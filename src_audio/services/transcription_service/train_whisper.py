import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 1. Configuration
model_name_or_path = "openai/whisper-large-v3" # RTX 3090 can handle large-v3 easily with LoRA
dataset_name = "leduckhai/MultiMed-ST"
output_dir = "./whisper-large-v3-medical-lora"

# 2. Load Assets
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name_or_path, language="English", task="transcribe")

# 3. Load and Prepare Dataset
ds = load_dataset(dataset_name)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Using num_proc to speed up preprocessing in WSL
ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=4)

# 4. Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Load Model with LoRA (Optimized for RTX 3090)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name_or_path, 
    load_in_8bit=True, 
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)

# 6. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2, # Effective batch size of 16
    learning_rate=1e-4,
    warmup_steps=100,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

# 7. Train
trainer.train()

# 8. Save the adapter
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)