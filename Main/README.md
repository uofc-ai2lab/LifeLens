Ignore the main folder as this will be used for later, you want to only be in the visual processing folder. in there we have two main folders, Classification and Object detection.
These are the two components, within each is a readme that explains the thought process for the code. This is more like mental notes for me so you dont need to worry too much about it.

How to run the components is as follows:
This PR has two model components separately:

Object detection script produces body-part crops and annotations.

Swin-Tiny classification script trains on an ImageFolder dataset with internal train/val(/test) splitting.

No unified pipeline is introduced here; that will follow in a subsequent PR but added some file structure for now

Object Detection Test Steps

Ensure sample images exist in the ImageSamples (if they don't, the readme has ALL dataset and model links, use those)
Run detection (adjust model/source as needed) OR you can just hit play but ensure that the arg parameters (in def parse_args()) matches what you want!

Note that the default max images is 1000 but you can change this depending on how much you want to run and how much is in your sample set

If you want to run the cmd command then run:

python VisualProcessing/ObjectDetection/detect_body_parts.py \
  --model MnLgt/yolo-human-parse \
  --source VisualProcessing/ObjectDetection/Priv_personpart/ImageSamples \
  --out VisualProcessing/ObjectDetection/outputs \
  --max-images 10
Verify outputs:
Crops: (if default) is under ./ObjectDetection/outputs/crops
Annotated images: (if default) is under ./ObjectDetection/outputs/annotated

Classification Test Steps

Folder should already be prepared from the readme file and places as:
VisualProcessing/Classification/ImageData/images/Wound_dataset/
Abrasion/
Bruise/
Burn/
Cut/
Laceration/
Stab_wound/
Normal skin/

you can just press play to run the code (same deal as before) or you can enter the cmd:

python VisualProcessing/Classification/ClassificationModels/simple_train_swin_tiny.py \
  --data-dir VisualProcessing/Classification/ImageData/images/Wound_dataset \
  --epochs 5 --val-ratio 0.2 --split-seed 42
Verify artifacts (must have already run model)
Checkpoint: check that best_swin_tiny_patch4_window7_224.pt exists
-> if it you cant see it under experiments/checkpoints/simple, run

Get-ChildItem -Recurse -File "experiments/checkpoints/simple/best_swin_tiny_patch4_window7_224.pt"
as it may not show up in the explorer for a bit

Then to see the previews run the predict_show_images file (names will be changed later"

Previews: under experiments/previews ensure that the image previews of the prediction is there
Metrics JSON (if created): same folder
Confusion matrix PNG: same folder

Environment
Requires: torch, torchvision, timm, numpy, scikit-learn, matplotlib.