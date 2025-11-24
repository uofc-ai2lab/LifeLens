param(
  [string]$TrainConfigEff = "experiments\config.efficientnet_b3.yaml",
  [string]$TrainConfigConv = "experiments\config.convnext_tiny.yaml",
  [string]$TrainConfigSwin = "experiments\config.swin_tiny.yaml"
)

$ErrorActionPreference = "Stop"

function Run-Train($cfg) {
  Write-Host "\n=== Training with config: $cfg ===\n"
  python src\training\train_multilabel.py --config $cfg
}

Run-Train $TrainConfigEff
Run-Train $TrainConfigConv
Run-Train $TrainConfigSwin

Write-Host "\nAll runs finished. Check experiments\checkpoints\*\metrics.json for results."
