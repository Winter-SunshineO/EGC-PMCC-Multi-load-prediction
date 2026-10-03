# Reviewer reproduction order

This public package keeps the executable recipe and removes unpublished narrative experiment material. Reviewers should use the wrapper below instead of reconstructing commands from archived notes.

```bash
python reproduce.py check
python reviewer_recipe.py --run-root runs/reviewer --device cuda:0 --execute
```

For a short interface test:

```bash
python reviewer_recipe.py --run-root runs/smoke --device cuda:0 --smoke --execute
```

The wrapper starts with the integrity check and then follows the dependency graph encoded in `configs/run_specs.json`. It creates an isolated run root, records the active interpreter and device, validates prerequisites, performs validation-only analysis before test export, and writes test and supplementary reports only after their required training steps finish.

Use `--resume` after an interruption. Do not reuse a run root after changing source code, recipe files, data, or device. Run roots are intentionally excluded from the public repository.

The public package does not include unpublished plans, historical result tables, or model-selection commentary. The machine-readable recipe is retained solely for reviewer verification.
