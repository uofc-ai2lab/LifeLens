import torch
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 1. Paths
adapter_model_dir = "./whisper-large-v3-medical-lora" # Your training output
merged_model_dir = "./whisper-medical-final"         # Where the final model goes

# 2. Load the Config and the Base Model
config = PeftConfig.from_pretrained(adapter_model_dir)
base_model = WhisperForConditionalGeneration.from_pretrained(
    config.base_model_name_or_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)

# 3. Load the Adapter onto the Base Model
model = PeftModel.from_pretrained(base_model, adapter_model_dir)

# 4. Merge and Unload
log.info("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

# 5. Save the standalone model and processor
merged_model.save_pretrained(merged_model_dir)
processor = WhisperProcessor.from_pretrained(adapter_model_dir)
processor.save_pretrained(merged_model_dir)

log.success(f"Standalone model saved to {merged_model_dir}")