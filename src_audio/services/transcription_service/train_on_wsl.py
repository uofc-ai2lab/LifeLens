import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

# 1. Load Pre-trained Parakeet TDT
model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

# 2. Update Model Config for your manifests
model.setup_training_data(train_data_config={
    'manifest_filepath': 'multimed_train_manifest.json',
    'sample_rate': 16000,
    'batch_size': 4, # Increase to 8 or 16 if you have >12GB VRAM
    'shuffle': True,
})

model.setup_validation_data(val_data_config={
    'manifest_filepath': 'multimed_test_manifest.json',
    'sample_rate': 16000,
    'batch_size': 4,
    'shuffle': False,
})

# 3. Setup Trainer (Optimized for WSL2 GPU)
trainer = pl.Trainer(
    devices=1,
    accelerator='gpu',
    precision='16-mixed', # Uses Tensor Cores for 2x speed
    max_epochs=5,         # Start with 5 for fine-tuning
    accumulate_grad_batches=2,
    enable_checkpointing=True
)

# 4. Train
print("Starting Medical Fine-tuning...")
trainer.fit(model)

# 5. Export Final Model
model.save_to("medical_parakeet_final.nemo")
print("Model saved as: medical_parakeet_final.nemo")