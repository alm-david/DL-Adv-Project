"""Encode the .npy images from prepare_data.py with SD-VAE and DINOv2-S.

For every <split>_images.npy under DATA_DIR, writes next to it:
    <split>_vae.npy        float32 [N, 8, 32, 32]   SD-VAE (ft-mse) posterior: channels 0-3 mean, 4-7 std
    <split>_dinov2_s.npy   float16 [N, 257, 384]    DINOv2-S/14 tokens: 0 = class token, 1..256 = patch tokens

Usage:
    python encode_data.py data [--batch-size 64]

Splits whose outputs already exist are skipped, so an interrupted run can be restarted.
Model weights download on first use (Hugging Face + torch.hub), so the first run needs internet.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_models(device):
    print("Loading SD-VAE (stabilityai/sd-vae-ft-mse)...", flush=True)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    print("Loading DINOv2-S (dinov2_vits14)...", flush=True)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    print("Models loaded.", flush=True)
    return vae.to(device).eval(), dino.to(device).eval()


@torch.no_grad()
def encode(images, vae, dino):
    """images: uint8 [B, 256, 256, 3] on device -> (vae [B, 8, 32, 32], dino [B, 257, 384])."""
    x = images.permute(0, 3, 1, 2).float() / 255  # [B, 3, 256, 256] in [0, 1]

    # SD-VAE expects [-1, 1]. Keep mean and std so training can sample z = mean + std * eps (as REPA/REG do).
    dist = vae.encode(x * 2 - 1).latent_dist
    vae_out = torch.cat([dist.mean, dist.std], dim=1)

    # DINOv2 preprocessing as in REPA/REG: ImageNet normalisation, bicubic resize 256 -> 224 (16x16 patches of 14 px).
    d = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
    d = F.interpolate(d, size=224, mode="bicubic", align_corners=False)
    feats = dino.forward_features(d)
    tokens = torch.cat([feats["x_norm_clstoken"][:, None], feats["x_norm_patchtokens"]], dim=1)
    return vae_out, tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="folder with the output of prepare_data.py")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: no GPU found, this will be very slow", flush=True)
    vae, dino = load_models(device)

    paths = sorted(Path(args.data_dir).rglob("*_images.npy"))
    print(f"Found {len(paths)} image files under {args.data_dir}: " + ", ".join(str(p) for p in paths), flush=True)
    if not paths:
        raise SystemExit("Nothing to encode: no *_images.npy files found")

    for img_path in paths:
        split = img_path.name[: -len("_images.npy")]
        out_vae = img_path.with_name(f"{split}_vae.npy")
        out_dino = img_path.with_name(f"{split}_dinov2_s.npy")
        if out_vae.exists() and out_dino.exists():
            print(f"{img_path}: already encoded, skipping")
            continue

        images = np.load(img_path, mmap_mode="r")
        n = len(images)
        print(f"\n{img_path}: {images.shape} {images.dtype}, batch size {args.batch_size}", flush=True)
        t0 = time.time()
        # Write to .partial files and rename at the end, so a half-finished file is never mistaken for a done one.
        tmp_vae, tmp_dino = out_vae.with_suffix(".partial"), out_dino.with_suffix(".partial")
        vae_arr = np.lib.format.open_memmap(tmp_vae, mode="w+", dtype=np.float32, shape=(n, 8, 32, 32))
        dino_arr = np.lib.format.open_memmap(tmp_dino, mode="w+", dtype=np.float16, shape=(n, 257, 384))

        for start in range(0, n, args.batch_size):
            batch = torch.from_numpy(np.array(images[start : start + args.batch_size])).to(device)
            v, t = encode(batch, vae, dino)
            vae_arr[start : start + len(batch)] = v.cpu().numpy()
            dino_arr[start : start + len(batch)] = t.half().cpu().numpy()
            done = start + len(batch)
            if start == 0:  # sanity check on the first batch: catches wrong scaling / broken models early
                mean, std = v[:, :4], v[:, 4:]
                print(
                    f"  first batch: VAE mean {mean.mean():.3f} +- {mean.std():.3f} "
                    f"(x0.18215 -> std {mean.std() * 0.18215:.3f}, should be ~1), VAE std avg {std.mean():.4f}, "
                    f"DINO token norm {t.norm(dim=-1).mean():.1f}",
                    flush=True,
                )
                if device == "cuda":
                    print(f"  GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            if not (torch.isfinite(v).all() and torch.isfinite(t).all()):
                raise RuntimeError(f"NaN/inf in outputs for images {start}..{done - 1} of {img_path}")
            if done % (50 * args.batch_size) < args.batch_size or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / rate
                print(f"  {done}/{n} ({100 * done / n:.0f}%)  {rate:.0f} img/s  elapsed {elapsed / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)

        vae_arr.flush()
        dino_arr.flush()
        del vae_arr, dino_arr
        os.replace(tmp_vae, out_vae)
        os.replace(tmp_dino, out_dino)
        print(f"  wrote {out_vae} and {out_dino} in {(time.time() - t0) / 60:.1f} min", flush=True)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
