"""Statistical definitions copied from the v7 offline analysis."""
import numpy as np
import pandas as pd
from scipy.stats import t
SECONDARY_TEST_FAMILY = "secondary"
SEEDS = (2021, 2023, 2024)
DETERMINISTIC = ("persistence", "daily_persistence", "weekly_persistence")


def origin_loss(y, pred, mask):
    valid = mask.astype(bool)
    count = valid.sum(axis=(1, 2))
    if np.any(count == 0):
        raise ValueError("empty origins")
    return np.where(valid, np.abs(y - pred), 0.0).sum(axis=(1, 2)) / count

def summarize(frame, groups, metrics, expected_seeds=SEEDS):
    records = []
    for key, data in frame.groupby(groups, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        if data.seed.duplicated().any():
            raise ValueError("duplicated physical seed in summary")
        deterministic = row.get("method") in DETERMINISTIC
        expected = {2021} if deterministic else set(expected_seeds)
        if set(data.seed) != expected:
            raise ValueError(f"incomplete seed group: {row}")
        row.update(n_effective=len(expected), seed_values=";".join(map(str, sorted(expected))))
        for metric in metrics:
            values = data[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"undefined {metric} in {row}; do not silently drop seeds")
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            half = float(t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else np.nan
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                        f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half})
            row[f"{metric}_mean_sd"] = f"{mean:.4f} +/- {sd:.4f}" if len(values) > 1 else f"{mean:.4f} (n=1; SD/CI N/A)"
        records.append(row)
    return pd.DataFrame(records)

def moving_block_estimates(diff, block_length, replicates, seed):
    """Same noncircular/truncated draws as the frozen scalar sampler."""
    diff = np.asarray(diff, dtype=np.float64)
    n = len(diff)
    if diff.ndim != 1 or not np.isfinite(diff).all() or not 1 <= block_length <= n or replicates < 1:
        raise ValueError("invalid MBB input")
    rng = np.random.default_rng(seed)
    full, remainder = divmod(n, block_length)
    pieces = full + bool(remainder)
    sums = np.r_[0.0, np.cumsum(diff)]
    estimates = np.empty(replicates)
    # Block sums avoid allocating full resampled series; draws retain row-major order.
    for offset in range(0, replicates, 256):
        size = min(256, replicates - offset)
        starts = rng.choice(n - block_length + 1, size=(size, pieces))
        total = (sums[starts[:, :full] + block_length] - sums[starts[:, :full]]).sum(axis=1)
        if remainder:
            total += sums[starts[:, -1] + remainder] - sums[starts[:, -1]]
        estimates[offset:offset + size] = total / n
    return estimates

def mbb_seed_aggregate(diffs, horizon, replicates=10000, seed=20240905):
    if len(diffs) != 3 or len({len(d) for d in diffs}) != 1:
        raise ValueError("MBB requires three aligned seeds")
    derived_seed = seed + horizon
    observed = float(np.mean([d.mean() for d in diffs]))
    estimates = np.mean([moving_block_estimates(d, horizon, replicates, derived_seed + 17 * i)
                         for i, d in enumerate(diffs)], axis=0)
    null = np.mean([moving_block_estimates(d - d.mean(), horizon, replicates, derived_seed + 1001 + 17 * i)
                   for i, d in enumerate(diffs)], axis=0)
    return dict(mean_diff=observed, ci95_low=float(np.quantile(estimates, .025)),
                ci95_high=float(np.quantile(estimates, .975)),
                p_raw=float((1 + np.count_nonzero(null <= observed)) / (replicates + 1)),
                n_windows=len(diffs[0]), n_seeds=3, block_length=horizon,
                replicates=replicates, bootstrap_seed=derived_seed)

def holm_adjust(
    reports: list[dict[str, object]],
    key: str,
    family_field: str = "test_family",
    output_key: str = "p_holm",
) -> None:
    """Apply Holm within each pre-registered family, never across families."""
    families: dict[str, list[int]] = {}
    for index, report in enumerate(reports):
        family = str(report.get(family_field, SECONDARY_TEST_FAMILY))
        try:
            p_value = float(report.get(key, np.nan))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric p-value in {family}: {report.get(key)!r}") from exc
        if not np.isfinite(p_value):
            raise ValueError(f"non-finite p-value in {family}: {report.get(key)!r}")
        families.setdefault(family, []).append(index)
    for family, indices in families.items():
        indexed = [
            (index, float(reports[index][key]))
            for index in indices
        ]
        indexed.sort(key=lambda item: item[1])
        adjusted = 0.0
        total = len(indexed)
        for rank, (index, p_value) in enumerate(indexed):
            adjusted = max(adjusted, min(1.0, p_value * (total - rank)))
            reports[index][output_key] = float(adjusted)
