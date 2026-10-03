# Public v7 data bundle

This directory contains the versioned model-input tables used by the public reviewer recipe. The raw upstream merged table is intentionally excluded from the repository.

The training scripts read `dataset_input.csv`, `dataset_labels.csv`, and `label_valid_mask.csv`. The remaining sidecar files preserve the data contract, normalization metadata, and integrity checks required by the runners.

The bundle was generated with the repository preprocessing script and is verified by `python reproduce.py check`. Rebuilding from upstream data requires the providers' permission and should use a separate output directory.
