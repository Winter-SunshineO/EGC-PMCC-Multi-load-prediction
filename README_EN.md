# EGC-PMCC 5.0 - Multi-Load Forecasting Reproduction Package

[Chinese](README.md) | **English**

This repository is a public reviewer package containing the model source code, versioned processed data, dependencies, and a fixed execution entry point. Raw upstream data, trained weights, and local run caches are excluded.

## Release scope

To protect unpublished research details before formal publication, the repository omits unpublished experiment plans, historical result tables, model-selection records, reviewer follow-up material, and historical analysis scripts. The retained machine-readable recipe exists only so reviewers can retrain the registered workflow, export test predictions, and regenerate the statistical outputs. It should not be treated as a publication result archive.

Use `reviewer_recipe.py` as the only public workflow entry point. It executes integrity checks, training, validation analysis, test export, offline analysis, supplementary analysis, and efficiency checks in dependency order. New outputs are written to an isolated `runs/` directory.

See the [reviewer reproduction order](docs/REPRODUCTION_ORDER.md) for the procedural contract.

## Installation

```bash
git clone https://github.com/Winter-SunshineO/EGC-PMCC-Multi-load-prediction.git EGC-PMCC5.0
cd EGC-PMCC5.0
conda create -n egc-pmcc-v7 python=3.11.15 -y
conda activate egc-pmcc-v7
python -m pip install -r requirements-public.txt
python reproduce.py check
```

For GPU execution, install a PyTorch CUDA build compatible with your driver and replace `--device cuda:0` with an available device when necessary. CPU execution is supported, but runtime and numerical results may differ.

## Reviewer reproduction order

Full run:

```bash
python reviewer_recipe.py --run-root runs/reviewer --device cuda:0 --execute
```

Short interface check:

```bash
python reviewer_recipe.py --run-root runs/smoke --device cuda:0 --smoke --execute
```

After an interruption, rerun the same command with `--resume`. Check disk space, GPU memory, and the CUDA environment before a formal run; the complete workflow may take substantial time. Every step writes execution records and output hashes, and failed runs are never overwritten.

For debugging, individual actions can be called through `reproduce.py`, but keep the order defined in `reviewer_recipe.py` and use a new `--run-root` whenever code, configuration, data, or device changes.

## Data

The processed v7 bundle is in `data/preprocessed_forecasting_v7/`. The training entry point loads the input table, target table, validity mask, and preprocessing metadata automatically. The raw merged table is not redistributed; terms for the processed tables are documented in [DATA_LICENSE.md](DATA_LICENSE.md). Data integrity is checked using `configs/data_checksums.json`.

## Outputs

Each run root stores environment fingerprints, execution records, checkpoints, validation outputs, test exports, and offline summaries. Run roots are ignored by `.gitignore` and should not be committed to GitHub.

## Code and licenses

The main public components include `reproduce.py`, `reviewer_recipe.py`, `train.py`, `deep_energy_baselines.py`, `energy_domain_baselines.py`, `torch_cfc.py`, `cfc_pci_heads.py`, `quantile_heads.py`, `metrics.py`, and `scripts/release_*.py`. Third-party sources and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); project code is covered by [LICENSE](LICENSE).

The paper has not yet been formally published. The complete citation and unpublished experimental interpretation will be added when appropriate.
