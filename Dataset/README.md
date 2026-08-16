# SEMICON v2 Dual Dataset

Verified Kaggle dataset:

<https://www.kaggle.com/datasets/omkadam05/semicon-v2-dual-dataset>

The multi-gigabyte dataset archives are intentionally not stored in this GitHub repository. Download the published data from Kaggle and extract it outside the repository or into a locally ignored data directory.

## SEMICON_SR_2X_v2

- Purpose: 2x super-resolution.
- Paired samples: 4,158.
- HR resolution: 1024x768.
- LR resolution: 512x384.
- Degradation: bicubic 2x downsampling only.
- Split: 2,910 train, 831 validation, and 417 test images.

## SEMICON_Noise_v2

- Purpose: same-resolution denoising and restoration.
- Paired samples: 4,158.
- Clean HR resolution: 1024x768.
- Degraded input resolution: 1024x768.
- Split: 2,910 train, 831 validation, and 417 test images.
- Documented degradation families: linewidth/CD bias, corner rounding, beam astigmatism, barrel distortion, pincushion distortion, vignetting, gamma adjustment, charging streaks, speckle noise, salt-and-pepper noise, and Gaussian noise.

Both datasets use the same deterministic source-level split identities. See `Dataset_Audit_Report.pdf` for the retained integrity, geometry, degradation, and training-readiness audit.

Expected downloaded archive names:

```text
SEMICON_SR_2X_v2.zip
SEMICON_Noise_v2.zip
```
