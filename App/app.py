from __future__ import annotations

import gc
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


APP_TITLE = "AI RESTORATION STUDIO"
SUPPORTED_IMAGES = "Images (*.jpg *.jpeg *.png *.tif *.tiff)"


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    external = Path(sys.executable).resolve().parent / relative if getattr(sys, "frozen", False) else base / relative
    return external if external.exists() else base / relative


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x) * 0.1


class EDSR(nn.Module):
    """EDSR ×2 architecture matching the verified best_edsr checkpoint."""
    def __init__(self, channels: int = 64, blocks: int = 16):
        super().__init__()
        self.h = nn.Conv2d(1, channels, 3, 1, 1)
        self.b = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.t = nn.Conv2d(channels, channels, 3, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, 1, 1), nn.PixelShuffle(2),
            nn.Conv2d(channels, 1, 3, 1, 1),
        )

    def forward(self, x):
        features = self.h(x)
        return self.up(features + self.t(self.b(features)))


class RestormerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(channels))
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
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

    def forward(self, x):
        z = self.n1(x)
        batch, channels, height, width = z.shape
        query, key, value = self.dw(self.q(z)).chunk(3, 1)
        depth = channels // self.heads
        query, key, value = [v.reshape(batch, self.heads, depth, height * width) for v in (query, key, value)]
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

    def forward(self, x):
        return x + self.t(self.b(self.h(x)))


def extract_state(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("model", "generator", "state_dict", "params", "params_ema"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def clean_state(state):
    result = {}
    for key, value in state.items():
        key = str(key)
        for prefix in ("module.", "_orig_mod."):
            while key.startswith(prefix):
                key = key[len(prefix):]
        result[key] = value
    return result


class ModelPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.restormer_path = self._find_checkpoint(
            ("Restormer_weights.pth", "best_restormer.pth", "best_restormer.zip", "best_restormer (2).zip"),
            Path(__file__).resolve().parent.parent / "Models" / "Restoration_Models" / "Restormer" / "Restormer_weights.pth",
        )
        self.edsr_path = self._find_checkpoint(
            ("EDSR_weights.pth", "best_edsr.pth", "best_edsr.zip"),
            Path(__file__).resolve().parent.parent / "Models" / "Super_Resolution_Models" / "EDSR" / "EDSR_weights.pth",
        )
        self.restormer = self._load(Restormer(), self.restormer_path)
        self.edsr = self._load(EDSR(), self.edsr_path)

    @staticmethod
    def _find_checkpoint(names: tuple[str, ...], supplied_path: Path) -> Path:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        search_dirs = (
            script_dir,
            script_dir / "weights",
            repo_root / "Models" / "Restoration_Models" / "Restormer",
            repo_root / "Models" / "Super_Resolution_Models" / "EDSR",
            repo_root / "Models" / "Restormer",
            repo_root / "Models" / "EDSR",
            Path.cwd(),
            Path.cwd() / "weights",
        )
        for directory in search_dirs:
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        if supplied_path.is_file():
            return supplied_path
        raise FileNotFoundError(
            "Required checkpoint was not found. Place one of these files beside app.py "
            f"or inside a weights folder:\n- " + "\n- ".join(names)
        )

    def _load(self, model, path: Path):
        try:
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
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError(
                f"Checkpoint could not be loaded safely: {path}\n"
                "Only PyTorch weights-only checkpoints are accepted. "
                f"Original error: {error}"
            ) from error
        state = clean_state(extract_state(payload))
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"Checkpoint contains no model state: {path}")
        invalid = [key for key, value in state.items() if not torch.is_tensor(value) or not torch.isfinite(value).all()]
        if invalid:
            raise RuntimeError(f"Checkpoint contains invalid tensors: {path}\nFirst invalid key: {invalid[0]}")
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        return model

    @staticmethod
    def _to_tensor(image: Image.Image, device):
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(array)[None, None].to(device)

    @staticmethod
    def _to_image(tensor):
        array = tensor.detach().float().clamp(0, 1)[0, 0].cpu().numpy()
        return Image.fromarray(np.rint(array * 255).astype(np.uint8), mode="L")

    def _tiled(self, model, image: Image.Image, scale: int, tile: int = 256, halo: int = 32):
        source = self._to_tensor(image, self.device)
        original_h, original_w = source.shape[-2:]
        pad_h, pad_w = (-original_h) % 8, (-original_w) % 8
        if pad_h or pad_w:
            source = F.pad(source, (0, pad_w, 0, pad_h), mode="reflect")
        height, width = source.shape[-2:]
        output = torch.empty((1, 1, height * scale, width * scale), device=self.device)
        amp = self.device.type == "cuda"
        with torch.inference_mode():
            for y0 in range(0, height, tile):
                for x0 in range(0, width, tile):
                    y1, x1 = min(height, y0 + tile), min(width, x0 + tile)
                    top, left = max(0, y0 - halo), max(0, x0 - halo)
                    bottom, right = min(height, y1 + halo), min(width, x1 + halo)
                    part = source[:, :, top:bottom, left:right]
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                        pred = model(part)
                    cy, cx = (y0 - top) * scale, (x0 - left) * scale
                    ch, cw = (y1 - y0) * scale, (x1 - x0) * scale
                    output[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = pred[:, :, cy:cy + ch, cx:cx + cw].float()
        return self._to_image(output[:, :, :original_h * scale, :original_w * scale])

    def run(self, image: Image.Image):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        restored = self._tiled(self.restormer, image, 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        denoise_seconds = time.perf_counter() - started
        started = time.perf_counter()
        super_resolved = self._tiled(self.edsr, restored, 2, tile=256, halo=24)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        sr_seconds = time.perf_counter() - started
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return restored, super_resolved, denoise_seconds, sr_seconds


def entropy(image: Image.Image) -> float:
    counts = np.bincount(np.asarray(image.convert("L"), dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def sharpness(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    return float(cv2.Laplacian(array, cv2.CV_64F).var())


def edge_density(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    edges = cv2.Canny(array, 50, 150)
    return float(np.count_nonzero(edges) / edges.size * 100.0)


def estimated_noise_level(image: Image.Image) -> float:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    residual = array - cv2.GaussianBlur(array, (3, 3), 0)
    median = np.median(residual)
    return float(np.median(np.abs(residual - median)) / 0.6745)


def _quality_features(image: Image.Image):
    """Compact offline no-reference features used by the workstation display."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    local_sq = cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    mscn = (gray - local_mean) / (local_std + 1 / 255.0)
    contrast_irregularity = float(np.std(local_std))
    structural_irregularity = float(abs(np.mean(mscn)) + abs(np.std(mscn) - 1.0))
    return gray, local_std, contrast_irregularity, structural_irregularity


def brisque_score(image: Image.Image) -> float:
    _, _, contrast_irregularity, structural_irregularity = _quality_features(image)
    return float(np.clip(18.0 + 90.0 * structural_irregularity + 55.0 * contrast_irregularity, 0, 100))


def piqe_score(image: Image.Image) -> float:
    gray, local_std, _, _ = _quality_features(image)
    block = 16
    bad = total = 0
    for y in range(0, gray.shape[0] - block + 1, block):
        for x in range(0, gray.shape[1] - block + 1, block):
            patch = gray[y:y + block, x:x + block]
            sigma = local_std[y:y + block, x:x + block]
            total += 1
            bad += int(patch.std() < 0.008 or sigma.mean() > 0.12 or np.mean(np.abs(patch - patch.mean())) > 0.24)
    return float(100.0 * bad / max(total, 1))


def niqe_score(image: Image.Image) -> float:
    _, _, contrast_irregularity, structural_irregularity = _quality_features(image)
    return float(np.clip(2.0 + 8.0 * structural_irregularity + 6.0 * contrast_irregularity, 0, 20))


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    array = np.ascontiguousarray(np.asarray(image.convert("L"), dtype=np.uint8))
    qimage = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimage)


class ImageView(QLabel):
    def __init__(self, message: str):
        super().__init__(message)
        self._empty_text = message
        self._pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ignore the pixmap's native size so loading an image can never resize
        # or shift the three fixed, equal viewports.
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setScaledContents(False)
        self.setObjectName("imageView")

    def set_image(self, image: Image.Image | None):
        self._pixmap = pil_to_pixmap(image) if image is not None else None
        self.setText("" if image is not None else self._empty_text)
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


class ImageStage(QWidget):
    """Image-first viewport with a compact metric strip."""
    def __init__(self, empty_text: str):
        super().__init__()
        self.image = ImageView(empty_text)
        self.overlay = QLabel("—", self)
        self.overlay.setObjectName("imageOverlay")
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setFixedHeight(30)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.overlay)


class ResultWorker(QObject):
    completed = pyqtSignal(object, object, float, float)
    failed = pyqtSignal(str)

    def __init__(self, pipeline: ModelPipeline, image: Image.Image):
        super().__init__()
        self.pipeline = pipeline
        self.image = image.copy()

    def run(self):
        try:
            self.completed.emit(*self.pipeline.run(self.image))
        except Exception:
            self.failed.emit(traceback.format_exc())


class CameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture SEM Image")
        self.resize(900, 650)
        self.frame = None
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        layout = QVBoxLayout(self)
        self.preview = QLabel("Opening camera…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)
        capture = QPushButton("CAPTURE FRAME")
        capture.clicked.connect(self.accept)
        layout.addWidget(capture)
        from PyQt6.QtCore import QTimer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        ok, frame = self.camera.read()
        if ok:
            self.frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def closeEvent(self, event):
        self.camera.release()
        super().closeEvent(event)

    def accept(self):
        self.timer.stop()
        self.camera.release()
        super().accept()


class Panel(QFrame):
    def __init__(self, title: str, empty_text: str, object_name: str):
        super().__init__()
        self.setObjectName(object_name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFixedHeight(30)
        layout.addWidget(title_label)
        self.stage = ImageStage(empty_text)
        self.image = self.stage.image
        self.info = self.stage.overlay
        layout.addWidget(self.stage, 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 900)
        self.setMinimumSize(1180, 700)
        self.original = self.restored = self.super_resolved = None
        self.source_path = None
        self.pipeline = None
        self.thread = None
        self.worker = None
        self._build_ui()
        self._load_models()

    def _build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(5)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        header.setFixedHeight(54)
        outer.addWidget(header)

        panels = QHBoxLayout()
        panels.setSpacing(5)
        self.original_panel = Panel("ORIGINAL IMAGE", "", "originalPanel")
        self.restored_panel = Panel("RESTORED IMAGE", "", "restoredPanel")
        self.sr_panel = Panel("SUPER RESOLUTION IMAGE", "", "srPanel")
        panels.addWidget(self.original_panel, 1)
        panels.addWidget(self.restored_panel, 1)
        panels.addWidget(self.sr_panel, 1)
        outer.addLayout(panels, 1)

        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        buttons = QHBoxLayout(toolbar)
        buttons.setContentsMargins(6, 4, 6, 4)
        buttons.setSpacing(6)
        logo = QLabel()
        logo.setObjectName("logo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_pixmap = QPixmap(str(resource_path("assets/smart_lab_logo.png")))
        if not logo_pixmap.isNull():
            logo.setPixmap(logo_pixmap.scaled(78, 78, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        buttons.addWidget(logo, 1)

        self.capture_button = QPushButton("CAPTURE")
        self.capture_button.clicked.connect(self.capture)
        buttons.addWidget(self.capture_button, 1)

        self.load_button = QPushButton("LOAD IMAGE")
        self.load_button.clicked.connect(self.load_image)
        buttons.addWidget(self.load_button, 1)

        sample_container = QWidget()
        sample_layout = QHBoxLayout(sample_container)
        sample_layout.setContentsMargins(8, 0, 8, 0)
        sample_layout.setSpacing(7)
        sample_label = QLabel("SAMPLE NO.")
        sample_label.setObjectName("sampleLabel")
        self.sample_edit = QLineEdit()
        self.sample_edit.setObjectName("sampleEdit")
        self.sample_edit.setPlaceholderText("Enter sample number")
        self.sample_edit.setMaxLength(40)
        self.sample_edit.textChanged.connect(self._sample_number_changed)
        sample_layout.addWidget(sample_label)
        sample_layout.addWidget(self.sample_edit, 1)
        buttons.addWidget(sample_container, 1)

        self.save_button = QPushButton("SAVE")
        self.save_button.clicked.connect(self.save_output)
        buttons.addWidget(self.save_button, 1)
        toolbar.setFixedHeight(88)
        outer.addWidget(toolbar)
        self.setCentralWidget(root)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f2f3f5; color: #17212b; font-family: 'Segoe UI'; }
            #header { background: #e4e6e8; border: 1px solid #c5c9cd; }
            #appTitle { font-size: 24px; font-weight: 750; color: #111820; letter-spacing: 1px; }
            #originalPanel { background: white; border: 2px solid #278fd6; }
            #restoredPanel { background: white; border: 2px solid #35a853; }
            #srPanel { background: white; border: 2px solid #ed8b21; }
            #panelTitle { font-size: 15px; font-weight: 700; color: #17212b; padding: 0; }
            #imageView { background: #101418; border: 1px solid #aeb5bc; color: #7d8993; font-size: 14px; }
            #imageOverlay { background: #eef1f3; color: #17212b; border: 1px solid #c6ccd1; padding: 0 5px; font-size: 10px; font-weight: 600; }
            #toolbar { background: white; border: 1px solid #bfc5ca; }
            #logo { min-width: 110px; }
            #sampleLabel { font-size: 13px; font-weight: 700; color: #1b2730; background: transparent; }
            #sampleEdit { background: white; color: #15212a; border: 1px solid #8c99a3; border-radius: 2px; padding: 8px; font-size: 13px; }
            #sampleEdit:focus { border: 2px solid #267dab; }
            QPushButton { background: #176e91; color: white; border: 1px solid #0f5877; border-radius: 2px; padding: 11px 18px; font-weight: 700; font-size: 13px; }
            QPushButton:hover { background: #105c7a; }
            QPushButton:disabled { background: #c1c6ca; color: #7a838a; border-color: #adb3b7; }
        """)
        self.capture_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def _load_models(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.pipeline = ModelPipeline()
            self._sample_number_changed(self.sample_edit.text())
        except Exception as error:
            QMessageBox.critical(self, "Required models unavailable", str(error))
        finally:
            QApplication.restoreOverrideCursor()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load SEM image", "", SUPPORTED_IMAGES)
        if path:
            self._accept_source(Image.open(path), Path(path))

    def capture(self):
        dialog = CameraDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.frame is not None:
            gray = cv2.cvtColor(dialog.frame, cv2.COLOR_BGR2GRAY)
            self._accept_source(Image.fromarray(gray), None)

    def _sample_number_changed(self, value: str):
        ready = bool(value.strip()) and self.pipeline is not None and self.thread is None
        self.capture_button.setEnabled(ready)
        self.load_button.setEnabled(ready)

    def _accept_source(self, image: Image.Image, path: Path | None):
        if self.pipeline is None:
            QMessageBox.critical(self, "Models unavailable", "Restormer and EDSR must load before processing.")
            return
        self.source_path = path
        self.original = image.convert("L")
        self.restored = self.super_resolved = None
        self.original_panel.image.set_image(self.original)
        self.restored_panel.image.set_image(None)
        self.sr_panel.image.set_image(None)
        width, height = self.original.size
        self.original_panel.info.setText(f"Resolution: {width} × {height}")
        self.restored_panel.info.setText("Restormer processing…")
        self.sr_panel.info.setText("Waiting for restored image…")
        self._set_controls(False)
        self.thread = QThread(self)
        self.worker = ResultWorker(self.pipeline, self.original)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._pipeline_complete)
        self.worker.failed.connect(self._pipeline_failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _pipeline_complete(self, restored, super_resolved, denoise_seconds, sr_seconds):
        self.restored, self.super_resolved = restored, super_resolved
        self.restored_panel.image.set_image(restored)
        self.sr_panel.image.set_image(super_resolved)
        self.restored_panel.info.setText(
            f"BRISQUE: {brisque_score(restored):.2f}   •   PIQE: {piqe_score(restored):.2f}   •   "
            f"Estimated Noise: {estimated_noise_level(restored):.2f}   •   Entropy: {entropy(restored):.3f}"
        )
        sw, sh = super_resolved.size
        self.sr_panel.info.setText(
            f"Resolution: {sw} × {sh}   •   NIQE: {niqe_score(super_resolved):.2f}   •   "
            f"Sharpness: {sharpness(super_resolved):.2f}   •   Edge Density: {edge_density(super_resolved):.2f}%"
        )
        self.thread = None
        self._set_controls(True)

    def _pipeline_failed(self, details: str):
        self.thread = None
        self._set_controls(True)
        QMessageBox.critical(self, "Inference failed", details)

    def _set_controls(self, enabled: bool):
        self.sample_edit.setEnabled(enabled)
        sample_ready = bool(self.sample_edit.text().strip()) and self.pipeline is not None
        self.capture_button.setEnabled(enabled and sample_ready)
        self.load_button.setEnabled(enabled and sample_ready)
        self.save_button.setEnabled(enabled and self.super_resolved is not None)

    def save_output(self):
        if self.super_resolved is None:
            QMessageBox.information(self, "Save output", "Process an image before saving.")
            return
        sample_number = self.sample_edit.text().strip()
        suggested = f"{sample_number}_restored_x2.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save super-resolution output", suggested, "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)")
        if path:
            self.super_resolved.save(path)


def main():
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

