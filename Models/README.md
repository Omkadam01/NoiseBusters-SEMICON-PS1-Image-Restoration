# Trained Models

This directory contains the verified training notebooks and checkpoints available from the NoiseBusters experiments.

## Final selected models

| Stage | Model | Task | Reason for selection |
|---|---|---|---|
| 1 | Restormer | Same-resolution denoising/restoration | Designed for image restoration and able to model spatially varying degradation while preserving semiconductor structure. |
| 2 | EDSR | 2x super-resolution | Stable reconstruction-oriented SR model with a practical accuracy, runtime, and deployment trade-off. |

The deployed pipeline is:

```text
SEM input -> Restormer -> EDSR x2 -> enhanced output
```

The deployable notebooks and checkpoints are stored in `Super_Resolution_Models/EDSR/` and `Restoration_Models/Restormer/`. The application and evaluation utility resolve their default checkpoints from these folders.

## Core experiments

| Model | Task family | Role |
|---|---|---|
| SRCNN | Super-resolution | Lightweight convolutional baseline |
| EDSR | Super-resolution | Final selected SR model |
| RCAN | Super-resolution | Channel-attention benchmark |
| ESRGAN | Super-resolution | Perceptual/GAN benchmark |
| MIRNet-v2 | Denoising/restoration | Multi-scale restoration benchmark |
| MPRNet | Denoising/restoration | Multi-stage progressive benchmark |
| NAFNet | Denoising/restoration | Efficient nonlinear-activation-free benchmark |
| Restormer | Denoising/restoration | Final selected restoration model |

## Additional benchmark

SwinIR is retained as an additional transformer-based super-resolution benchmark because its notebook, checkpoint, metrics, and test outputs are available. It is not part of the deployed application pipeline.

## Reproducibility notes

- Each model folder contains its available training notebook and corresponding trained checkpoint.
- Final comparisons should use a harmonized test protocol because notebook implementations may differ in metric details.
- Checkpoints and the executable are assigned to Git LFS through the root `.gitattributes` file.
- Dataset splits must always be read from the published dataset manifests; do not create new random splits.
