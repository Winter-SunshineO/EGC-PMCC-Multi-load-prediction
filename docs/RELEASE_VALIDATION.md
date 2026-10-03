# Public release validation

The public tree is validated by the same checks available to reviewers:

```bash
python -m compileall -q reproduce.py reviewer_recipe.py scripts configs
python reproduce.py check
python reviewer_recipe.py --run-root runs/smoke --device cuda:0 --smoke --execute
```

The smoke workflow checks process launching, data verification, model output registration, offline analysis, and the supplementary entry points. It is an interface check and is not a paper result. A formal run must use a new run root and the full reviewer recipe.

The raw upstream merged table, historical experiment notes, result tables, checkpoints, and local caches are not part of the public tree. Processed-data terms are documented in [DATA_LICENSE.md](../DATA_LICENSE.md).
