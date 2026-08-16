# Evaluation Outputs

This directory contains retained evidence from the available model experiments.

## Benchmark_Results

Model-specific folders contain the available final metrics, per-image metrics, training histories, learning curves, and visual comparison grids. `MODEL_METRICS_COMPARISON.csv` consolidates reported model values for convenient comparison.

These files are intended for reproducibility and review. Metrics across independently authored notebooks should be interpreted carefully unless their preprocessing, crop policy, border handling, SSIM implementation, and inference protocol have been harmonized.

## Test_Outputs

The folder contains representative model-labelled test comparison figures. Filenames identify the model that generated each result. These outputs support qualitative inspection of semiconductor edges, textures, noise suppression, and possible hallucinated detail.

For new evaluation runs, use `Evaluation/inference.py`. It writes restored images and reference metrics without modifying the published datasets.
