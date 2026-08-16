# NoiseBusters - SEMICON PS1 Image Restoration

Final submission repository for **Team NoiseBusters** in the **SEMICON India Hackathon 2026, Problem Statement 1**.

The project restores degraded MEMS/SEM microscopy images with a fixed two-stage pipeline:

```text
Input SEM image
    -> Restormer denoising/restoration
    -> EDSR 2x super-resolution
    -> enhanced output
```

## Project contents

- Reproducible training notebooks and trained checkpoints.
- Standalone command-line inference and reference evaluation.
- Benchmark histories, metrics, plots, and representative test outputs.
- Dataset documentation and technical audit report.
- Final technical report and hackathon presentation.
- Native Windows desktop application and installer definition.

## Dataset

The project uses the **SEMICON v2 Dual Dataset**, derived from 4,158 NFFA-Europe MEMS SEM source images:

- `SEMICON_SR_2X_v2`: 512x384 LR to 1024x768 HR, with bicubic 2x downsampling.
- `SEMICON_Noise_v2`: 1024x768 degraded input to 1024x768 clean target, covering 11 documented degradation families.
- Shared source-level split: 2,910 train, 831 validation, and 417 test images.

Download the datasets from Kaggle:

<https://www.kaggle.com/datasets/omkadam05/semicon-v2-dual-dataset>

The multi gigabyte archives are intentionally excluded from GitHub. See [`Dataset/README.md`](Dataset/README.md) and [`Dataset/Dataset_Audit_Report.pdf`](Dataset/Dataset_Audit_Report.pdf).

## Repository structure

```text
NoiseBusters-SEMICON-PS1-Image-Restoration/
├── README.md
├── requirements.txt
├── Dataset/
│   ├── README.md
│   └── Dataset_Audit_Report.pdf
├── Models/
│   ├── README.md
│   ├── Restoration_Models/
│   ├── Super_Resolution_Models/
│   └── Final_Selected_Models/
│       ├── EDSR/
│       └── Restormer/
├── Evaluation/
│   └── inference.py
├── Outputs/
│   ├── README.md
│   ├── Benchmark_Results/
│   └── Test_Outputs/
├── Documentation/
│   ├── Final_Project_Report.pdf
│   └── SEMICON_Project_Presentation.pptx
└── App/
    ├── app.py
    ├── AI_Restoration_Studio.exe
    ├── AI_Restoration_Studio.iss
    └── assets/
```

## Installation

Python 3.10-3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU acceleration is used automatically when a compatible PyTorch CUDA installation is available. The included requirements provide the project packages but do not force a specific CUDA build.

## Evaluation and inference

Run the deployed Restormer-to-EDSR pipeline:

```powershell
python Evaluation\inference.py --input sample.png --mode full
```

Run an individual stage:

```powershell
python Evaluation\inference.py --input noisy_images --mode denoise
python Evaluation\inference.py --input lr_images --mode sr
```

Evaluate against an explicitly supplied ground-truth directory:

```powershell
python Evaluation\inference.py --input test_inputs --ground-truth test_gt --mode denoise
```

Outputs are written as PNG images together with `per_image_metrics.csv` and `summary.json`. PSNR, SSIM, MAE, MSE, and RMSE are calculated only when matching ground truth is supplied.

## Desktop application

Run from source:

```powershell
python App\app.py
```

Alternatively, launch:

```text
App/AI_Restoration_Studio.exe
```

The native application implements the fixed `Restormer -> EDSR x2` workflow. The included executable is unsigned; Windows may display a SmartScreen warning after downloading it.

## Models

Eight core experiments are retained for reproducibility and comparison: SRCNN, EDSR, RCAN, ESRGAN, MIRNet-v2, MPRNet, NAFNet, and Restormer. A SwinIR benchmark is also retained because its verified notebook, checkpoint, and outputs are available.

The final deployed models are:

- **Restormer** for same-resolution denoising and restoration.
- **EDSR** for 2x super-resolution.

See [`Models/README.md`](Models/README.md) for task assignments and selection rationale.

## GitHub large-file setup

This repository contains checkpoint files and a Windows executable that must be stored with Git LFS.

```powershell
git lfs install
git lfs track "*.pth" "*.pt" "*.exe"
git add .gitattributes
```

Do not add dataset ZIP files, build directories, virtual environments, caches, or generated installer outputs.

## Documentation

- [`Documentation/Final_Project_Report.pdf`](Documentation/Final_Project_Report.pdf)
- [`Documentation/SEMICON_Project_Presentation.pptx`](Documentation/SEMICON_Project_Presentation.pptx)

## GitHub readiness

The repository is organized for direct GitHub upload. Dataset archives are provided through Kaggle, generated files are ignored, and large checkpoints/application binaries are assigned to Git LFS. Before public release, confirm Git LFS quota and all applicable dataset, architecture, and checkpoint licensing terms.

## Attribution and licensing

The source imagery originates from NFFA-Europe. Users must follow the authoritative source-data and Kaggle dataset terms. Model architectures and checkpoints may also be governed by their originating project licenses. No verified project-level license was available in the supplied artifacts.
