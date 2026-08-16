"""Standalone SEMICON inference and optional reference-evaluation utility.

Examples:
    python Evaluation/inference.py --input sample.png --mode full
    python Evaluation/inference.py --input test_inputs --ground-truth test_gt --mode denoise
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESTORMER = REPO_ROOT / "Models" / "Final_Selected_Models" / "Restormer" / "Restormer_weights.pth"
DEFAULT_EDSR = REPO_ROOT / "Models" / "Final_Selected_Models" / "EDSR" / "EDSR_weights.pth"


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * 0.1


class EDSR(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 16):
        super().__init__()
        self.h = nn.Conv2d(1, channels, 3, 1, 1)
        self.b = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.t = nn.Conv2d(channels, channels, 3, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.h(x)
        return self.up(features + self.t(self.b(features)))


class RestormerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.w, self.b).permute(0, 3, 1, 2)


class RestormerBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.n1 = RestormerNorm(channels)
        self.q = nn.Conv2d(channels, channels * 3, 1)
        self.dw = nn.Conv2d(channels * 3, channels * 3, 3, 1, 1, groups=channels * 3)
        self.p = nn.Conv2d(channels, channels, 1)
        self.t = nn.Parameter(torch.ones(heads, 1, 1))
        self.n2 = RestormerNorm(channels)
        self.f1 = nn.Conv2d(channels, channels * 4, 1)
        self.fd = nn.Conv2d(channels * 4, channels * 4, 3, 1, 1, groups=channels * 4)
        self.f2 = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.n1(x)
        batch, channels, height, width = z.shape
        query, key, value = self.dw(self.q(z)).chunk(3, 1)
        depth = channels // self.heads
        query, key, value = [item.reshape(batch, self.heads, depth, height * width) for item in (query, key, value)]
        attention = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1) * self.t).softmax(-1)
        x = x + self.p((attention @ value).reshape(batch, channels, height, width))
        left, right = self.fd(self.f1(self.n2(x))).chunk(2, 1)
        return x + self.f2(F.gelu(left) * right)


class Restormer(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 12):
        super().__init__()
        self.h = nn.Conv2d(1, channels, 3, 1, 1)
        self.b = nn.Sequential(*[RestormerBlock(channels) for _ in range(blocks)])
        self.t = nn.Conv2d(channels, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.t(self.b(self.h(x)))


def extract_state(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("model", "generator", "state_dict", "params", "params_ema"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def clean_state(state):
    cleaned = {}
    for key, value in state.items():
        key = str(key)
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def load_model(model: nn.Module, checkpoint: Path, device: torch.device) -> nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    numpy_safe_types = [
        np._core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.uint32)),
        type(np.dtype(np.uint8)),
        type(np.dtype(np.int32)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.bool_)),
    ]
    with torch.serialization.safe_globals(numpy_safe_types):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = clean_state(extract_state(payload))
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"No model state found in {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, None].to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1)[0, 0].cpu().numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="L")


def tiled_predict(model, image: Image.Image, device, scale: int, tile: int, halo: int) -> Image.Image:
    source = image_to_tensor(image, device)
    original_h, original_w = source.shape[-2:]
    pad_h, pad_w = (-original_h) % 8, (-original_w) % 8
    if pad_h or pad_w:
        source = F.pad(source, (0, pad_w, 0, pad_h), mode="reflect")
    height, width = source.shape[-2:]
    output = torch.empty((1, 1, height * scale, width * scale), device=device)
    amp = device.type == "cuda"
    with torch.inference_mode():
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                y1, x1 = min(height, y0 + tile), min(width, x0 + tile)
                top, left = max(0, y0 - halo), max(0, x0 - halo)
                bottom, right = min(height, y1 + halo), min(width, x1 + halo)
                part = source[:, :, top:bottom, left:right]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    prediction = model(part)
                crop_y, crop_x = (y0 - top) * scale, (x0 - left) * scale
                crop_h, crop_w = (y1 - y0) * scale, (x1 - x0) * scale
                output[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = prediction[
                    :, :, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w
                ].float()
    return tensor_to_image(output[:, :, : original_h * scale, : original_w * scale])


def global_ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    mean_x, mean_y = reference.mean(), prediction.mean()
    var_x, var_y = reference.var(), prediction.var()
    covariance = ((reference - mean_x) * (prediction - mean_y)).mean()
    return float(((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)))


def reference_metrics(output: Image.Image, ground_truth: Image.Image) -> dict[str, float]:
    prediction = np.asarray(output.convert("L"), dtype=np.float64) / 255.0
    reference = np.asarray(ground_truth.convert("L"), dtype=np.float64) / 255.0
    if prediction.shape != reference.shape:
        raise ValueError(f"Output/GT shape mismatch: {prediction.shape} vs {reference.shape}")
    error = prediction - reference
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    return {
        "psnr": float("inf") if mse == 0 else float(10 * math.log10(1.0 / mse)),
        "ssim_global": global_ssim(reference, prediction),
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
    }


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED)


def find_ground_truth(root: Path | None, source: Path) -> Path | None:
    if root is None:
        return None
    if root.is_file():
        return root
    direct = root / source.name
    if direct.is_file():
        return direct
    matches = list(root.rglob(source.name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one ground-truth match for {source.name}; found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restormer denoising and EDSR 2x inference")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs"))
    parser.add_argument("--ground-truth", type=Path, help="Optional GT image or directory for reference metrics")
    parser.add_argument("--mode", choices=("denoise", "sr", "full"), default="full")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--restormer-weight", type=Path, default=DEFAULT_RESTORMER)
    parser.add_argument("--edsr-weight", type=Path, default=DEFAULT_EDSR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    restormer = load_model(Restormer(), args.restormer_weight, device) if args.mode in ("denoise", "full") else None
    edsr = load_model(EDSR(), args.edsr_weight, device) if args.mode in ("sr", "full") else None
    sources = collect_images(args.input)
    if not sources:
        raise RuntimeError("No supported input images were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sources:
        with Image.open(source) as handle:
            image = handle.convert("L")
        started = time.perf_counter()
        output = image
        if restormer is not None:
            output = tiled_predict(restormer, output, device, scale=1, tile=args.tile, halo=32)
        if edsr is not None:
            output = tiled_predict(edsr, output, device, scale=2, tile=args.tile, halo=24)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        destination = args.output_dir / f"{source.stem}_{args.mode}.png"
        output.save(destination)
        row = {"input": str(source), "output": str(destination), "mode": args.mode, "seconds": elapsed, "device": str(device)}
        gt_path = find_ground_truth(args.ground_truth, source)
        if gt_path is not None:
            with Image.open(gt_path) as handle:
                row.update(reference_metrics(output, handle.convert("L")))
            row["ground_truth"] = str(gt_path)
        rows.append(row)
        print(f"{source.name} -> {destination.name} ({elapsed:.3f}s)")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    numeric = [key for key in ("seconds", "psnr", "ssim_global", "mae", "mse", "rmse") if key in fieldnames]
    summary = {"images": len(rows), "mode": args.mode, "device": str(device)}
    for key in numeric:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
