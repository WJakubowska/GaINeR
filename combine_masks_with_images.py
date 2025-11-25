from pathlib import Path
import argparse
from PIL import Image

#!/usr/bin/env python3
"""
combine_masks_with_images.py

Load images from a folder and corresponding masks from another folder,
combine them by using the mask as the alpha channel, and save results
as PNGs with transparent background into an output "rgba" folder.

Defaults are set to the paths you provided; you can override via CLI.
"""

experiment_name = "matador"

DEFAULT_IMAGES_DIR = Path(f"renders/GaINeR/{experiment_name}/test/rgb")
DEFAULT_MASKS_DIR = Path(f"renders/GaINeR/{experiment_name}_mask/test/rgb")
DEFAULT_OUTPUT_DIR = Path(f"renders/GaINeR/{experiment_name}/rgba")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def find_mask_for_stem(masks_dir: Path, stem: str):
    for p in masks_dir.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def zero_out_transparent_pixels(img: Image.Image) -> Image.Image:
    """
    Ensure that pixels with alpha == 0 have RGB channels set to 0.
    Preserves the original alpha channel.
    """
    r, g, b, a = img.split()
    # Binary mask: 255 where alpha > 0, else 0
    mask_binary = a.point(lambda v: 255 if v > 0 else 0)
    rgb = Image.merge("RGB", (r, g, b))
    black = Image.new("RGB", img.size, (0, 0, 0))
    rgb_zeroed = Image.composite(rgb, black, mask_binary)
    return Image.merge("RGBA", (*rgb_zeroed.split(), a))


def combine_image_and_mask(image_path: Path, mask_path: Path, out_path: Path):
    img = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, resample=Image.BILINEAR)
    img.putalpha(mask)
    # Zero RGB where alpha == 0
    img = zero_out_transparent_pixels(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def main(images_dir: Path, masks_dir: Path, output_dir: Path):
    if not images_dir.exists():
        raise SystemExit(f"Images folder does not exist: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"Masks folder does not exist: {masks_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not files:
        print("No image files found in", images_dir)
        return

    for img_path in files:
        stem = img_path.stem
        mask_path = find_mask_for_stem(masks_dir, stem)
        if mask_path is None:
            print(f"Skipping {img_path.name}: no matching mask for stem '{stem}'")
            continue
        out_path = output_dir / f"{stem}.png"
        try:
            combine_image_and_mask(img_path, mask_path, out_path)
            print("Saved:", out_path)
        except Exception as e:
            print(f"Failed to process {img_path.name} with {mask_path.name}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Combine RGB images with masks to produce RGBA PNGs with transparency.")
    p.add_argument("--images", "-i", type=Path, default=DEFAULT_IMAGES_DIR, help="Folder with input images (RGB).")
    p.add_argument("--masks", "-m", type=Path, default=DEFAULT_MASKS_DIR, help="Folder with masks (grayscale or RGB).")
    p.add_argument("--out", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output folder to save RGBA PNGs.")
    args = p.parse_args()
    main(args.images, args.masks, args.out)