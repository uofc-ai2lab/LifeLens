import argparse
import os
from typing import Tuple, List

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
from PIL import Image, ImageDraw, ImageFont


def build_eval_dataset(data_root: str, image_size: int):
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_root, transform=val_transforms)
    return dataset


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name")
    class_names = ckpt.get("class_names")
    if model_name is None or class_names is None:
        raise ValueError("Checkpoint missing 'model_name' or 'class_names'.")
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, class_names


def annotate_image(img_path: str, text: str, out_path: str):
    # Load image and prepare an RGBA overlay for proper transparency handling
    base = Image.open(img_path).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Try to load a common font; fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Compute text size using modern Pillow API if available
    margin = 5
    padding = 4
    if hasattr(odraw, "textbbox"):
        left, top, right, bottom = odraw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
    else:
        # Fallback for older Pillow
        try:
            text_w, text_h = odraw.textsize(text, font=font)  # type: ignore[attr-defined]
        except Exception:
            # Last resort estimate
            text_w = int(8 * len(text))
            text_h = 16

    rect = [
        margin,
        margin,
        margin + text_w + 2 * padding,
        margin + text_h + 2 * padding,
    ]
    # Semi-transparent black rectangle
    odraw.rectangle(rect, fill=(0, 0, 0, 160))
    odraw.text((margin + padding, margin + padding), text, font=font, fill=(255, 255, 255, 255))

    annotated = Image.alpha_composite(base.convert("RGBA"), overlay)
    annotated.convert("RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Save a few annotated predictions for visual inspection")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder with class subfolders (ImageFolder)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint saved by simple trainers")
    parser.add_argument("--out-dir", type=str, default=os.path.join("experiments", "previews"))
    parser.add_argument("--count", type=int, default=12, help="Number of images to preview")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset", type=str, default="all", choices=["all", "train", "val"], help="Which subset to preview (reconstructed by seed/val-ratio)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio to reconstruct split if subset != all")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the split")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    dataset = build_eval_dataset(args.data_dir, args.img_size)
    ds_classes = dataset.classes
    model, ckpt_classes = load_model_from_checkpoint(args.checkpoint, device)

    # Mapping dataset labels to checkpoint labels by name
    name_to_idx_ckpt = {name: i for i, name in enumerate(ckpt_classes)}
    if set(ds_classes) != set(ckpt_classes):
        raise ValueError("Class names differ between dataset and checkpoint.")

    # Reconstruct subset indices if needed
    all_indices: List[int] = list(range(len(dataset.samples)))
    if args.subset == "all":
        selected_indices = all_indices
    else:
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(all_indices), generator=generator).tolist()
        num_val = int(len(all_indices) * args.val_ratio)
        val_indices = perm[:num_val]
        train_indices = perm[num_val:]
        selected_indices = train_indices if args.subset == "train" else val_indices

    selected_samples = [dataset.samples[i] for i in selected_indices]
    subset_ds = Subset(dataset, selected_indices)
    loader = DataLoader(subset_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    saved = 0
    idx_offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1)
            pred_conf = probs.gather(1, pred_idx.view(-1, 1)).squeeze(1)

            for i in range(images.size(0)):
                if saved >= args.count:
                    break
                img_path, ds_label_idx = selected_samples[idx_offset + i]
                true_name = ds_classes[int(ds_label_idx)]
                pred_name = ckpt_classes[int(pred_idx[i].item())]
                conf = float(pred_conf[i].item())
                correct = int(pred_name == true_name)
                caption = f"pred: {pred_name} ({conf:.2f}) | true: {true_name} | {'✓' if correct else '✗'}"

                base = os.path.splitext(os.path.basename(img_path))[0]
                out_path = os.path.join(args.out_dir, f"preview_{saved+1:02d}_{base}.jpg")
                annotate_image(img_path, caption, out_path)
                saved += 1

            idx_offset += images.size(0)
            if saved >= args.count:
                break

    print(f"Saved {saved} previews to {args.out_dir}")


if __name__ == "__main__":
    main()
