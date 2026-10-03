# Publishing the public reviewer package

The GitHub repository is [Winter-SunshineO/EGC-PMCC-Multi-load-prediction](https://github.com/Winter-SunshineO/EGC-PMCC-Multi-load-prediction). The public branch should contain source code, the processed data bundle, machine-readable execution recipes, licenses, and the bilingual README files.

Before pushing a release, run:

```bash
python -m compileall -q reproduce.py reviewer_recipe.py scripts configs
python reproduce.py check
git status --short
git diff --check
```

Do not commit run roots, checkpoints, prediction arrays, raw merged data, historical experiment notes, result tables, or local environment files. Keep data terms separate from the MIT code license. Use a new version tag after the paper and full experimental materials are ready for public release.
