"""Download Food-101 and Imagenette and save 256x256 centre crops as .npy.

Usage:
    python prepare_data.py OUT_DIR [--raw-dir DIR]

Output (in OUT_DIR, default: current directory):
    food101/{train,test}_images.npy      uint8 [N, 256, 256, 3]
    food101/{train,test}_labels.npy      int64 [N]
    food101/classes.txt                  one class name per line (label = line index)
    imagenette/{train,val}_...           same
Downloads go to --raw-dir (default OUT_DIR/raw), which is deleted at the end.

Load with:  x = np.load("OUT_DIR/food101/train_images.npy", mmap_mode="r")
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import Food101, Imagenette

SIZE = 256


def center_crop_arr(pil_image, image_size):
    """Centre crop from ADM (guided-diffusion), as used by DiT/SiT/REPA/REG."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


parser = argparse.ArgumentParser()
parser.add_argument("out_dir", nargs="?", default=".", help="where the .npy files go")
parser.add_argument("--raw-dir", help="where to download/extract (default: OUT_DIR/raw); deleted at the end")
args = parser.parse_args()
out_dir = Path(args.out_dir)
raw = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"

datasets = {
    ("food101", "train"): lambda: Food101(raw, split="train", download=True),
    ("food101", "test"): lambda: Food101(raw, split="test", download=True),
    # Some torchvision versions raise if the folder already exists, so only download when it's missing.
    ("imagenette", "train"): lambda: Imagenette(raw, split="train", download=not (raw / "imagenette2").exists()),
    ("imagenette", "val"): lambda: Imagenette(raw, split="val", download=not (raw / "imagenette2").exists()),
}

for (name, split), make in datasets.items():
    out = out_dir / name
    if (out / f"{split}_labels.npy").exists():  # labels are written last, so this split finished
        print(f"{name}/{split}: already done, skipping")
        continue
    ds = make()
    out.mkdir(parents=True, exist_ok=True)

    # Written straight to disk, so the whole array never has to fit in RAM.
    images = np.lib.format.open_memmap(
        out / f"{split}_images.npy", mode="w+", dtype=np.uint8, shape=(len(ds), SIZE, SIZE, 3)
    )
    labels = np.empty(len(ds), dtype=np.int64)
    for i, (img, label) in enumerate(ds):
        images[i] = center_crop_arr(img.convert("RGB"), SIZE)
        labels[i] = label
    images.flush()
    np.save(out / f"{split}_labels.npy", labels)

    classes = [c if isinstance(c, str) else c[0] for c in ds.classes]
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    print(f"{name}/{split}: {images.shape}")

shutil.rmtree(raw)
print(f"Deleted {raw} (archives and extracted originals)")
