#!/usr/bin/env python3
"""Prepare a leakage-safe, model-ready multi-load forecasting table.

Only hard-invalid values are removed from labels: non-finite/parser failures,
explicit sentinels, source quality flags when supplied, and violations of the
pre-registered physical bounds. Physically plausible load peaks, troughs, and
zero values are retained. Training-only IQR statistics are emitted as
诊断 information only; they never invalidate a label or alter a model input.
The input-history copy is filled causally, while the original cleaned labels
and their validity mask remain separate.

The default output contains three causal input target histories plus nine
training-ranked features (12 numeric columns total). Cleaned labels and their
validity mask are stored separately, so no imputed validation/test value can
become a reported target.

Example
-------
python scripts/preprocess_forecasting_data.py \
    --input data/merged_data21-23.csv \
    --output-dir data/preprocessed_forecasting_v6 \
    --feature-count 9 --selection-threshold 0.20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["KW", "CHWTON", "HTmmBTU"]
TIME_COLUMNS = ["Year", "Month", "Day", "Hour"]
DERIVED_FEATURES = [
    "Hour_sin", "Hour_cos", "DOW_sin", "DOW_cos", "Month_sin", "Month_cos",
    "DayOfYear_sin", "DayOfYear_cos", "WindDirection_sin", "WindDirection_cos",
    "Temperature_sq", "CDD18", "HDD18",
]
RAW_FEATURES = [
    "DOW", "Temperature", "Dew Point", "Relative Humidity",
    "Solar Zenith Angle", "Surface Albedo", "Pressure", "Precipitable Water",
    "Wind Direction", "Wind Speed", "GHG",
]
# Combined mmBTU is deliberately excluded: it is derived from the target loads.
DEFAULT_FEATURE_COUNT = 9
DEFAULT_SELECTION_THRESHOLD = 0.20
# IQR is retained as a train-only diagnostic, never as a cleaning rule.
DEFAULT_IQR_FACTOR = 1.5
DEFAULT_MIN_REGIME_RUN_HOURS = 1
SENTINEL_ABS_LIMIT = 1.0e8
# v7-only robust target envelope.  ``abs(value) > max(floor, factor * median)``
# is fitted on training-split hard-valid targets and applied to all three
# target channels.  The median is used deliberately: an upper quantile is
# estimated from the contaminated sample, so a single 5.8e5 reading raises the
# very bound meant to exclude it (measured: the 99.9th-percentile form produced
# a KW threshold of 613,382, i.e. above KW's own 577,201 outlier).
#
# Off by default so the v6 recipe stays reproducible bit-for-bit; ``None`` means
# the rule is disabled and the cleaning audit records the omission explicitly.
DEFAULT_TARGET_ENVELOPE_FACTOR: float | None = None
DEFAULT_TARGET_ENVELOPE_FLOOR = 1.0
# Value recorded in the cleaning audit when the envelope rule is disabled.
TARGET_ENVELOPE_DISABLED = "disabled_v6_hard_validity_only"
V6_SCHEMA_VERSION = 6
V6_PROTOCOL = "model_input_v6_hard_validity_causal_labels_separated"
# v7 = v6 hard validity plus the robust target envelope.  The model-column
# contract, split policy, sidecar layout, sequence policy, and z-score recipe
# are unchanged, so a v7 bundle is a drop-in replacement for the runners'
# ``--data`` argument.
V7_SCHEMA_VERSION = 7
V7_PROTOCOL = "model_input_v7_hard_validity_robust_target_envelope"
SUPPORTED_SCHEMA_VERSIONS = (V6_SCHEMA_VERSION, V7_SCHEMA_VERSION)
SCHEMA_PROTOCOLS = {V6_SCHEMA_VERSION: V6_PROTOCOL, V7_SCHEMA_VERSION: V7_PROTOCOL}
FORMAL_SEQUENCE_POLICY = "csv_sidecars_only_embargo_origins"
PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    # Load zeros are physically valid and are deliberately retained.
    "KW": (0.0, None),
    "CHWTON": (0.0, None),
    "HTmmBTU": (0.0, None),
    "Temperature": (-273.15, None),
    "Dew Point": (-273.15, None),
    "Relative Humidity": (0.0, 100.0),
    "Solar Zenith Angle": (0.0, 180.0),
    "Surface Albedo": (0.0, 1.0),
    "Pressure": (0.0, None),
    "Precipitable Water": (0.0, None),
    "Wind Direction": (0.0, 360.0),
    "Wind Speed": (0.0, None),
    "GHG": (0.0, None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/merged_data21-23.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed_forecasting_v6"),
        help="new immutable versioned output directory; it must not already exist",
    )
    parser.add_argument("--window", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--iqr-factor", type=float, default=DEFAULT_IQR_FACTOR)
    parser.add_argument("--min-regime-run-hours", type=int, default=DEFAULT_MIN_REGIME_RUN_HOURS, help=argparse.SUPPRESS)
    parser.add_argument("--selection-threshold", type=float, default=DEFAULT_SELECTION_THRESHOLD)
    parser.add_argument("--feature-count", type=int, default=DEFAULT_FEATURE_COUNT)
    parser.add_argument("--zero-target-policy", choices=["keep", "error"], default="keep", help="retain physical zero targets; error is available only as a diagnostic guard")
    parser.add_argument(
        "--schema-version",
        type=int,
        choices=list(SUPPORTED_SCHEMA_VERSIONS),
        default=V6_SCHEMA_VERSION,
        help="6 keeps hard-validity-only cleaning; 7 adds the robust target envelope",
    )
    parser.add_argument(
        "--target-envelope-factor",
        type=float,
        default=None,
        help=(
            "v7 robust target envelope: invalidate a target when "
            "abs(value) > max(floor, factor * train median abs); required for schema 7"
        ),
    )
    parser.add_argument(
        "--target-envelope-floor",
        type=float,
        default=DEFAULT_TARGET_ENVELOPE_FLOOR,
        help="absolute floor for the v7 robust target envelope",
    )
    # Formal protocol always uses CSV plus label sidecars and runner-generated origins.
    parser.add_argument("--write-sequences", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quality-flags-json", type=Path, default=None, help="optional JSON map of value columns to source quality flag rules")
    return parser.parse_args()


def sanitize_numeric(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert parser failures and extreme sentinel encodings to NaN."""
    out = frame.copy()
    counts: dict[str, int] = {}
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values.abs() >= SENTINEL_ABS_LIMIT)
        counts[column] = int(invalid.sum())
        out[column] = values.mask(invalid)
    return out, counts


def load_quality_flag_rules(path: Path | None, columns: list[str], frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load optional source-quality invalidation rules without guessing flags."""
    if path is None:
        return {}, {"source": None, "columns": {}, "available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quality flag mapping must be a JSON object")
    invalid_masks: dict[str, np.ndarray] = {}
    recorded: dict[str, object] = {}
    for value_column, rule in payload.items():
        if value_column not in columns:
            raise ValueError(f"quality rule names unknown value column: {value_column}")
        if not isinstance(rule, dict) or "column" not in rule:
            raise ValueError(f"quality rule for {value_column} must contain a flag column")
        flag_column = str(rule["column"])
        if flag_column not in frame.columns:
            raise ValueError(f"quality flag column is missing from raw input: {flag_column}")
        invalid_values = rule.get("invalid_values", [])
        if not isinstance(invalid_values, list):
            raise ValueError(f"invalid_values must be a list for {value_column}")
        flags = frame[flag_column]
        invalid_masks[value_column] = flags.isin(invalid_values).to_numpy(dtype=bool)
        recorded[value_column] = {"column": flag_column, "invalid_values": invalid_values}
    return invalid_masks, {"source": str(path), "columns": recorded, "available": bool(recorded)}


def hard_validity_masks(
    frame: pd.DataFrame,
    columns: list[str],
    quality_invalid: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Return the only values allowed to invalidate a label in v6."""
    quality_invalid = quality_invalid or {}
    masks: dict[str, np.ndarray] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        lower, upper = PHYSICAL_BOUNDS.get(column, (None, None))
        invalid = ~np.isfinite(values) | (np.abs(values) >= SENTINEL_ABS_LIMIT)
        if lower is not None:
            invalid |= values < float(lower)
        if upper is not None:
            invalid |= values > float(upper)
        if column in quality_invalid:
            invalid |= np.asarray(quality_invalid[column], dtype=bool)
        masks[column] = invalid
    return masks


def build_timestamp(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = sorted(set(TIME_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"timestamp fields missing: {missing}")

    out = frame.copy()
    timestamp = pd.to_datetime(
        out[TIME_COLUMNS].rename(columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}),
        errors="coerce",
    )
    if timestamp.isna().any():
        raise ValueError(f"invalid timestamp rows: {int(timestamp.isna().sum())}")
    out["timestamp"] = timestamp
    duplicate_count = int(out["timestamp"].duplicated().sum())
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    out = out.set_index("timestamp")
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="h")
    inserted = int(len(full_index) - len(out))
    out = out.reindex(full_index)
    out.index.name = "timestamp"
    # Sunday=1, Monday=2, ..., Saturday=7, matching the source convention.
    out["DOW"] = ((out.index.dayofweek + 1) % 7 + 1).astype(float)
    return out, {"duplicate_timestamps": duplicate_count, "inserted_missing_hours": inserted}


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.DatetimeIndex(out.index)
    hour = ts.hour.to_numpy(float)
    dow = out["DOW"].to_numpy(float)
    month = ts.month.to_numpy(float)
    day_of_year = ts.dayofyear.to_numpy(float)

    out["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["DOW_sin"] = np.sin(2.0 * np.pi * (dow - 1.0) / 7.0)
    out["DOW_cos"] = np.cos(2.0 * np.pi * (dow - 1.0) / 7.0)
    out["Month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    out["Month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    out["DayOfYear_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
    out["DayOfYear_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1.0) / 365.25)

    wind_direction = pd.to_numeric(out["Wind Direction"], errors="coerce")
    temperature = pd.to_numeric(out["Temperature"], errors="coerce")
    out["WindDirection_sin"] = np.sin(2.0 * np.pi * wind_direction / 360.0)
    out["WindDirection_cos"] = np.cos(2.0 * np.pi * wind_direction / 360.0)
    out["Temperature_sq"] = temperature**2
    out["CDD18"] = (temperature - 18.0).clip(lower=0.0)
    out["HDD18"] = (18.0 - temperature).clip(lower=0.0)
    return out


def fit_iqr_rules(train: pd.DataFrame, columns: list[str], factor: float) -> dict[str, dict[str, float | None]]:
    """Fit train-only IQR diagnostics and hard-valid imputation medians.

    The lower/upper IQR values are audit statistics only. They never invalidate
    labels or inputs. Physical bounds remain the only numeric range rules.
    """
    rules: dict[str, dict[str, float | None]] = {}
    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        physical_lower, physical_upper = PHYSICAL_BOUNDS.get(column, (None, None))
        if physical_lower is not None:
            valid &= values >= physical_lower
        if physical_upper is not None:
            valid &= values <= physical_upper
        fit_values = values[valid]
        if fit_values.empty:
            raise ValueError(f"cannot fit cleaning rule: no hard-valid training values for {column}")
        q1, q3 = fit_values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        soft_lower = float(q1 - factor * iqr)
        soft_upper = float(q3 + factor * iqr)
        if soft_lower > soft_upper:
            raise ValueError(f"incompatible soft cleaning bounds for {column}: {soft_lower} > {soft_upper}")
        rules[column] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": iqr,
            "lower": soft_lower,
            "upper": soft_upper,
            "replacement": float(fit_values.median()),
            "physical_lower": physical_lower,
            "physical_upper": physical_upper,
            "fit_valid_count": int(valid.sum()),
            "fit_hard_invalid_count": int((values.notna() & np.isfinite(values) & ~valid).sum()),
        }
    return rules


def initial_regime_state(columns: list[str]) -> dict[str, dict[str, int]]:
    """Return a compatibility state; v6 does not use IQR state transitions."""
    return {column: {"consecutive_soft_excursions": 0, "excursion_side": 0} for column in columns}


def fit_target_envelope_rules(
    train: pd.DataFrame,
    factor: float,
    floor: float,
) -> dict[str, dict[str, float]]:
    """Fit the v7 robust target envelope on training-split hard-valid labels.

    A target value is hard-invalid when ``abs(value) > max(floor, factor *
    median(abs(valid training values)))``.  Only the three target channels are
    covered: the rule exists to catch physically impossible load readings that
    slip under the fixed ``SENTINEL_ABS_LIMIT``, and applying it to exogenous
    weather or calendar features would invalidate legitimate extremes.

    The median has a 50% breakdown point, so a handful of corrupted rows cannot
    move the bound.  A quantile-based bound was measured to fail exactly here.
    """
    if factor is None or factor <= 0:
        raise ValueError("target envelope factor must be a positive number")
    envelope: dict[str, dict[str, float]] = {}
    for column in TARGETS:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        physical_lower, physical_upper = PHYSICAL_BOUNDS.get(column, (None, None))
        valid = np.isfinite(values)
        if physical_lower is not None:
            valid &= values >= float(physical_lower)
        if physical_upper is not None:
            valid &= values <= float(physical_upper)
        valid &= np.abs(values) < SENTINEL_ABS_LIMIT
        fit_values = np.abs(values[valid])
        if fit_values.size == 0:
            raise ValueError(f"cannot fit target envelope: no hard-valid training values for {column}")
        median = float(np.median(fit_values))
        threshold = max(float(floor), factor * median)
        envelope[column] = {
            "fit_valid_count": int(fit_values.size),
            "median_abs": median,
            "mean_abs": float(fit_values.mean()),
            "max_abs": float(fit_values.max()),
            "factor": float(factor),
            "floor": float(floor),
            "threshold": threshold,
            "max_to_median_ratio": float(fit_values.max() / median) if median > 0 else float("inf"),
        }
    return envelope


def apply_causal_cleaning_rules(
    frame: pd.DataFrame,
    rules: dict[str, dict[str, float | None]],
    min_regime_run_hours: int = DEFAULT_MIN_REGIME_RUN_HOURS,
    initial_state: dict[str, dict[str, int]] | None = None,
    quality_invalid: dict[str, np.ndarray] | None = None,
    target_envelope: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Apply v6 hard validity rules without removing plausible excursions.

    IQR bounds in ``rules`` are diagnostic only. Values are invalidated solely
    by non-finite/sentinel values, physical bounds, supplied source-quality
    flags, or (v7 only) the train-fitted robust target envelope. The argument
    names are retained for compatibility with older audit callers; no stateful
    soft-excursion decision is made.
    """
    del min_regime_run_hours, initial_state
    out = frame.copy()
    quality_invalid = quality_invalid or {}
    target_envelope = target_envelope or {}
    audit: dict[str, dict[str, int]] = {}
    for column, rule in rules.items():
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        sentinel = finite & (np.abs(values) >= SENTINEL_ABS_LIMIT)
        physical = np.zeros(len(values), dtype=bool)
        lower = rule.get("physical_lower")
        upper = rule.get("physical_upper")
        if lower is not None:
            physical |= finite & (values < float(lower))
        if upper is not None:
            physical |= finite & (values > float(upper))
        quality = np.asarray(quality_invalid.get(column, np.zeros(len(values), dtype=bool)), dtype=bool)
        if quality.shape != values.shape:
            raise ValueError(f"quality mask shape mismatch for {column}: {quality.shape} != {values.shape}")
        # v7 robust envelope: catches the surviving member of a corruption burst
        # that sits below the fixed sentinel ceiling.
        envelope = np.zeros(len(values), dtype=bool)
        if column in target_envelope:
            threshold = float(target_envelope[column]["threshold"])
            envelope = finite & (np.abs(values) > threshold)
        invalid = ~finite | sentinel | physical | quality | envelope
        cleaned = values.copy()
        cleaned[invalid] = np.nan
        out[column] = cleaned
        diagnostic = finite & ~invalid & (
            (values < float(rule["lower"])) | (values > float(rule["upper"]))
        )
        audit[column] = {
            "input_missing_or_nonfinite": int(np.sum(~finite)),
            "explicit_sentinel_invalid": int(np.sum(sentinel)),
            "hard_physical_invalid": int(np.sum(physical)),
            "source_quality_invalid": int(np.sum(quality)),
            "target_envelope_invalid": int(np.sum(envelope)),
            "iqr_diagnostic_excursion_count": int(np.sum(diagnostic)),
            "iqr_removed_count": 0,
            # Compatibility fields retained for existing audit readers.
            "soft_excursion_total": int(np.sum(diagnostic)),
            "soft_provisional_removed": 0,
            "soft_confirmed_retained": int(np.sum(diagnostic)),
            "ordinary_retained": int(np.sum(finite & ~invalid & ~diagnostic)),
            "valid_retained": int(np.sum(~invalid)),
        }
    return out, audit, initial_regime_state(list(rules))


def apply_iqr_rules(
    frame: pd.DataFrame,
    rules: dict[str, dict[str, float | None]],
    min_regime_run_hours: int = DEFAULT_MIN_REGIME_RUN_HOURS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compatibility wrapper; returns hard-invalid counts only."""
    cleaned, audit, _ = apply_causal_cleaning_rules(frame, rules, min_regime_run_hours)
    return cleaned, {column: values["hard_physical_invalid"] for column, values in audit.items()}


def apply_zero_target_policy(frame: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict[str, int]]:
    out = frame.copy()
    counts: dict[str, int] = {}
    for column in TARGETS:
        values = pd.to_numeric(out[column], errors="coerce")
        zeros = values.eq(0)
        counts[column] = int(zeros.sum())
        if policy == "error" and zeros.any():
            raise ValueError(f"zero target values found in {column}: {int(zeros.sum())}")
        # Zero is a valid physical load value in v6. Never turn it into a
        # missing label merely because it is zero.
        out[column] = values
    return out, counts


def impute_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    rules: dict[str, dict[str, float | None]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Apply one causal imputation policy to all chronological splits.

    The state is carried forward through the chronological stream. The only
    fallback value is the training-fitted median; no later row is used to fill
    an earlier row.
    """
    tr, va, te = train.copy(), validation.copy(), test.copy()
    columns = list(rules)
    missing_before = {column: int(pd.concat([tr, va, te])[column].isna().sum()) for column in columns}
    for column in columns:
        replacement = float(rules[column]["replacement"])
        tr[column] = tr[column].ffill().fillna(replacement)
        state = float(tr[column].iloc[-1]) if len(tr) else replacement
        va[column] = va[column].ffill().fillna(state).fillna(replacement)
        state = float(va[column].iloc[-1]) if len(va) else state
        te[column] = te[column].ffill().fillna(state).fillna(replacement)
    missing_after = {column: int(pd.concat([tr, va, te])[column].isna().sum()) for column in columns}
    return tr, va, te, {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "policy": "causal forward fill in chronological order; leading gaps use the training-fitted median",
        "fit_split": "training split only",
        "apply_splits": ["training", "validation", "test"],
        "train": "causal forward fill, then training median for leading gaps",
        "validation_test": "causal forward fill seeded by the preceding split state, then training median",
    }


def correlation_screen(train: pd.DataFrame, candidates: list[str], threshold: float) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    for feature in candidates:
        for target in TARGETS:
            x = pd.to_numeric(train[feature], errors="coerce")
            y = pd.to_numeric(train[target], errors="coerce")
            pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                pearson, spearman = np.nan, np.nan
            else:
                pearson = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))
                spearman = float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
            rows.append({
                "target": target, "feature": feature, "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson) if np.isfinite(pearson) else np.nan,
                "abs_spearman": abs(spearman) if np.isfinite(spearman) else np.nan,
                "n_valid": int(len(pair)),
            })
    table = pd.DataFrame(rows)
    table["selected_by_threshold"] = table[["abs_pearson", "abs_spearman"]].max(axis=1) >= threshold
    scores = table.groupby("feature").agg(
        mean_abs_correlation=("abs_pearson", "mean"),
        mean_abs_spearman=("abs_spearman", "mean"),
        min_abs_correlation=("abs_pearson", "min"),
        threshold_pass_count=("selected_by_threshold", "sum"),
    )
    scores["screen_score"] = scores[["mean_abs_correlation", "mean_abs_spearman"]].mean(axis=1)
    scores = scores.sort_values(["screen_score", "min_abs_correlation"], ascending=False)
    table = table.merge(scores[["screen_score", "threshold_pass_count"]], left_on="feature", right_index=True)
    table["rank"] = table["screen_score"].rank(method="min", ascending=False).astype(int)
    eligible = scores[scores["threshold_pass_count"] > 0]
    selected = list(eligible.index)
    return table.sort_values(["rank", "target"]), selected


def make_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window: int,
    horizon: int,
    labels: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    targets = frame[TARGETS].to_numpy(dtype=np.float32)
    count = len(frame) - window - horizon + 1
    if count <= 0:
        empty_x = np.empty((0, window, len(feature_columns)), np.float32)
        empty_y = np.empty((0, horizon, len(TARGETS)), np.float32)
        empty_mask = None if labels is None else np.empty((0, horizon, len(TARGETS)), np.float32)
        return empty_x, empty_y, empty_mask
    x = np.stack([values[index:index + window] for index in range(count)])
    if labels is None:
        y = np.stack([targets[index + window:index + window + horizon] for index in range(count)])
        return x, y, None
    label_values = labels[TARGETS].to_numpy(dtype=np.float32)
    mask_values = np.isfinite(label_values).astype(np.float32)
    y = np.stack([label_values[index + window:index + window + horizon] for index in range(count)])
    mask = np.stack([mask_values[index + window:index + window + horizon] for index in range(count)])
    return x, y, mask


def fit_zscore(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats = {}
    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            raise ValueError(f"cannot fit z-score parameters: no finite training values for {column}")
        mean, std = float(values.mean()), float(values.std(ddof=0))
        stats[column] = {"mean": mean, "std": std if std >= 1e-8 else 1.0}
    return stats


def apply_zscore(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = frame.copy()
    for column, values in stats.items():
        out[column] = (out[column] - values["mean"]) / values["std"]
    return out


def select_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = frame.reset_index().rename(columns={"index": "timestamp"})
    return out[["timestamp", *TARGETS, *feature_columns]]


def causal_preprocess_core(
    raw: pd.DataFrame,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    iqr_factor: float = DEFAULT_IQR_FACTOR,
    selection_threshold: float = DEFAULT_SELECTION_THRESHOLD,
    feature_count: int = DEFAULT_FEATURE_COUNT,
    zero_target_policy: str = "keep",
    target_envelope_factor: float | None = DEFAULT_TARGET_ENVELOPE_FACTOR,
    target_envelope_floor: float = DEFAULT_TARGET_ENVELOPE_FLOOR,
) -> dict[str, object]:
    """Run the fit/apply portion without writing files for behavioural audits."""
    raw_numeric = [column for column in raw.columns if column not in TIME_COLUMNS]
    raw, _ = sanitize_numeric(raw, raw_numeric)
    indexed, _ = build_timestamp(raw)
    indexed = add_engineered_features(indexed)
    candidates = [column for column in RAW_FEATURES + DERIVED_FEATURES if column in indexed.columns]
    all_columns = TARGETS + candidates
    n = len(indexed)
    train_end = int(train_ratio * n)
    validation_end = int((train_ratio + validation_ratio) * n)
    train_raw = indexed.iloc[:train_end].copy()
    validation_raw = indexed.iloc[train_end:validation_end].copy()
    test_raw = indexed.iloc[validation_end:].copy()
    rules = fit_iqr_rules(train_raw, all_columns, iqr_factor)
    envelope = (
        fit_target_envelope_rules(train_raw, target_envelope_factor, target_envelope_floor)
        if target_envelope_factor is not None
        else {}
    )
    state = initial_regime_state(all_columns)
    train_clean, _, state = apply_causal_cleaning_rules(
        train_raw, rules, DEFAULT_MIN_REGIME_RUN_HOURS, state, target_envelope=envelope
    )
    validation_clean, _, state = apply_causal_cleaning_rules(
        validation_raw, rules, DEFAULT_MIN_REGIME_RUN_HOURS, state, target_envelope=envelope
    )
    test_clean, _, _ = apply_causal_cleaning_rules(
        test_raw, rules, DEFAULT_MIN_REGIME_RUN_HOURS, state, target_envelope=envelope
    )
    train_clean, _ = apply_zero_target_policy(train_clean, zero_target_policy)
    validation_clean, _ = apply_zero_target_policy(validation_clean, zero_target_policy)
    test_clean, _ = apply_zero_target_policy(test_clean, zero_target_policy)
    labels = {"train": train_clean[TARGETS].copy(), "validation": validation_clean[TARGETS].copy(), "test": test_clean[TARGETS].copy()}
    train_input, _, _, _ = impute_splits(train_clean, validation_clean, test_clean, rules)
    correlation_frame = train_input.copy()
    correlation_frame[TARGETS] = labels["train"]
    _, ranked = correlation_screen(correlation_frame, candidates, selection_threshold)
    selected = ranked[: min(feature_count, len(ranked))]
    model_columns = TARGETS + selected
    zsource = train_input.copy()
    zsource[TARGETS] = labels["train"]
    zstats = fit_zscore(zsource, model_columns)
    fingerprint = canonical_hash({
        "rules": rules,
        "target_envelope": envelope,
        "selected_features": selected,
        "zscore": zstats,
        "train_input": train_input[model_columns].round(12).to_dict(orient="list"),
    })
    return {
        "artifact_fingerprint": fingerprint,
        "rules": rules,
        "target_envelope": envelope,
        "selected_features": selected,
        "zscore": zstats,
    }


def build_output(args: argparse.Namespace, output_dir: Path) -> None:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not 0.5 <= args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in [0.5, 1.0)")
    if not 0.0 < args.validation_ratio < 1.0 - args.train_ratio:
        raise ValueError("--validation-ratio must leave a non-empty test split")
    if args.feature_count <= 0:
        raise ValueError("--feature-count must be positive")
    if args.iqr_factor <= 0 or args.selection_threshold < 0:
        raise ValueError("--iqr-factor must be positive and --selection-threshold non-negative")
    if args.min_regime_run_hours <= 0:
        raise ValueError("--min-regime-run-hours must be positive")
    if args.schema_version == V7_SCHEMA_VERSION and args.target_envelope_factor is None:
        raise ValueError("schema 7 requires --target-envelope-factor; the envelope is the v7 contract")
    if args.schema_version == V6_SCHEMA_VERSION and args.target_envelope_factor is not None:
        raise ValueError("--target-envelope-factor requires --schema-version 7; v6 output must stay hard-validity-only")
    if args.target_envelope_factor is not None and args.target_envelope_factor <= 0:
        raise ValueError("--target-envelope-factor must be positive")
    if args.target_envelope_floor <= 0:
        raise ValueError("--target-envelope-floor must be positive")
    if args.write_sequences:
        raise ValueError(
            "formal preprocessing forbids NPZ exports; use dataset_input.csv plus label sidecars and runner-generated embargo origins"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    schema_version = int(args.schema_version)
    protocol_name = SCHEMA_PROTOCOLS[schema_version]
    envelope_enabled = args.target_envelope_factor is not None

    raw_source = pd.read_csv(input_path)
    missing_targets = sorted(set(TARGETS) - set(raw_source.columns))
    if missing_targets:
        raise ValueError(f"missing target columns: {missing_targets}")
    raw_numeric = [column for column in raw_source.columns if column not in TIME_COLUMNS]
    quality_masks, quality_metadata = load_quality_flag_rules(args.quality_flags_json, raw_numeric, raw_source)
    raw, sentinel_counts = sanitize_numeric(raw_source, raw_numeric)
    source_quality_counts: dict[str, int] = {}
    for column, invalid in quality_masks.items():
        source_quality_counts[column] = int(np.sum(invalid))
        raw.loc[np.asarray(invalid, dtype=bool), column] = np.nan
    quality_metadata["invalid_counts"] = source_quality_counts
    quality_metadata["mapping_sha256"] = sha256(args.quality_flags_json.resolve()) if args.quality_flags_json else None
    indexed, time_audit = build_timestamp(raw)
    indexed = add_engineered_features(indexed)

    candidates = [column for column in RAW_FEATURES + DERIVED_FEATURES if column in indexed.columns]
    if not candidates:
        raise ValueError("no numeric candidate features available")
    all_columns = TARGETS + candidates
    n = len(indexed)
    train_end = int(args.train_ratio * n)
    validation_end = int((args.train_ratio + args.validation_ratio) * n)
    train_raw = indexed.iloc[:train_end].copy()
    validation_raw = indexed.iloc[train_end:validation_end].copy()
    test_raw = indexed.iloc[validation_end:].copy()

    rules = fit_iqr_rules(train_raw, all_columns, args.iqr_factor)
    target_envelope = (
        fit_target_envelope_rules(train_raw, args.target_envelope_factor, args.target_envelope_floor)
        if envelope_enabled
        else {}
    )
    regime_state = initial_regime_state(all_columns)
    train_clean, train_cleaning_audit, regime_state = apply_causal_cleaning_rules(
        train_raw, rules, args.min_regime_run_hours, regime_state, target_envelope=target_envelope
    )
    validation_clean, validation_cleaning_audit, regime_state = apply_causal_cleaning_rules(
        validation_raw, rules, args.min_regime_run_hours, regime_state, target_envelope=target_envelope
    )
    test_clean, test_cleaning_audit, final_regime_state = apply_causal_cleaning_rules(
        test_raw, rules, args.min_regime_run_hours, regime_state, target_envelope=target_envelope
    )

    train_clean, train_zero_counts = apply_zero_target_policy(train_clean, args.zero_target_policy)
    validation_clean, validation_zero_counts = apply_zero_target_policy(validation_clean, args.zero_target_policy)
    test_clean, test_zero_counts = apply_zero_target_policy(test_clean, args.zero_target_policy)
    # Keep cleaned labels separate from causal model inputs. Invalid labels are
    # never synthesized; only the input-history copy is imputed.
    label_frames = {
        "train": train_clean[TARGETS].copy(),
        "validation": validation_clean[TARGETS].copy(),
        "test": test_clean[TARGETS].copy(),
    }
    label_masks = {name: frame.notna().astype(np.float32) for name, frame in label_frames.items()}
    train_input, validation_input, test_input, imputation = impute_splits(
        train_clean, validation_clean, test_clean, rules
    )

    # Screen using causal training inputs but only observed training labels.
    correlation_frame = train_input.copy()
    correlation_frame[TARGETS] = label_frames["train"]
    correlations, ranked_features = correlation_screen(correlation_frame, candidates, args.selection_threshold)
    if not ranked_features:
        raise ValueError("no candidate features pass the training-only correlation threshold")
    selected_features = ranked_features[: min(args.feature_count, len(ranked_features))]
    if len(selected_features) < args.feature_count:
        raise ValueError(f"only {len(selected_features)} candidate features are available")
    model_columns = TARGETS + selected_features

    train = select_frame(train_input, selected_features)
    validation = select_frame(validation_input, selected_features)
    test = select_frame(test_input, selected_features)
    all_clean = pd.concat([train, validation, test], ignore_index=True)
    zscore_source = train_input.copy()
    zscore_source[TARGETS] = label_frames["train"]
    zstats = fit_zscore(zscore_source, model_columns)
    ztrain, zvalidation, ztest = [apply_zscore(frame, zstats) for frame in (train, validation, test)]

    # Direct input to train.py and baseline loaders: no timestamp, numeric only,
    # and the three forecast targets are first.
    dataset_input = all_clean[model_columns].copy()
    dataset_input_zscore = apply_zscore(dataset_input, zstats)
    dataset_labels = pd.concat([label_frames["train"], label_frames["validation"], label_frames["test"]], ignore_index=True)
    label_valid_mask = pd.concat([label_masks["train"], label_masks["validation"], label_masks["test"]], ignore_index=True)
    dataset_input.to_csv(output_dir / "dataset_input.csv", index=False, encoding="utf-8-sig")
    dataset_input_zscore.to_csv(output_dir / "dataset_input_zscore.csv", index=False, encoding="utf-8-sig")
    dataset_labels.to_csv(output_dir / "dataset_labels.csv", index=False, encoding="utf-8-sig", na_rep="nan")
    label_valid_mask.to_csv(output_dir / "label_valid_mask.csv", index=False, encoding="utf-8-sig")
    all_clean.to_csv(output_dir / "forecast_table.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(output_dir / "validation.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")
    for name, frame in (("train", ztrain), ("validation", zvalidation), ("test", ztest)):
        frame.to_csv(output_dir / f"{name}_zscore.csv", index=False, encoding="utf-8-sig")
    for name, frame in (("train", train_input), ("validation", validation_input), ("test", test_input)):
        frame[TARGETS].to_csv(output_dir / f"{name}_labels_input.csv", index=False, encoding="utf-8-sig")
        label_frames[name].to_csv(output_dir / f"{name}_labels.csv", index=False, encoding="utf-8-sig", na_rep="nan")
        label_masks[name].to_csv(output_dir / f"{name}_label_valid_mask.csv", index=False, encoding="utf-8-sig")

    sequence_info = None
    # Sequence tensors are deliberately absent from formal v6. Every runner
    # derives the same embargo origins from the CSV bundle.

    correlations.to_csv(output_dir / "train_correlations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame.from_dict(rules, orient="index").reset_index(names="feature").to_csv(
        output_dir / "train_fitted_iqr_bounds.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "schema_version": schema_version,
        "protocol": protocol_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "command": [sys.executable, *sys.argv],
            "working_directory": str(Path.cwd().resolve()),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "publication": "staged_then_atomic_directory_rename",
        },
        "input": str(input_path),
        "input_relative": str(args.input),
        "input_sha256": sha256(input_path),
        "output_dataset": "dataset_input.csv",
        "output_dataset_sha256": sha256(output_dir / "dataset_input.csv"),
        "label_dataset": "dataset_labels.csv",
        "label_dataset_sha256": sha256(output_dir / "dataset_labels.csv"),
        "label_valid_mask": "label_valid_mask.csv",
        "label_valid_mask_sha256": sha256(output_dir / "label_valid_mask.csv"),
        "raw_rows": int(len(raw)),
        "aligned_rows": int(n),
        "time_start": str(indexed.index.min()),
        "time_end": str(indexed.index.max()),
        "time_audit": time_audit,
        "split": {
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.validation_ratio,
            "train_end_exclusive": train_end,
            "validation_end_exclusive": validation_end,
            "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        },
        "targets": TARGETS,
        "target_units": {target: "kW" for target in TARGETS},
        "candidate_features": candidates,
        "selected_features": selected_features,
        "model_columns": model_columns,
        "model_input_dim": len(model_columns),
        "sentinel_abs_limit": SENTINEL_ABS_LIMIT,
        "physical_bounds": PHYSICAL_BOUNDS,
        "sentinel_or_nonfinite_counts": sentinel_counts,
        "iqr_diagnostic_factor": args.iqr_factor,
        "source_quality_flags": quality_metadata,
        "cleaning": {
            "rule": (
                "labels and inputs reject non-finite/parser failures, explicit sentinels, "
                "source quality failures, and pre-registered physical-bound violations; "
                f"training-only Q1 +/- {args.iqr_factor:g}*IQR bounds are diagnostic only and remove no rows"
                + (
                    "; the v7 robust target envelope additionally rejects target values above "
                    f"max({args.target_envelope_floor:g}, {args.target_envelope_factor:g} * train median abs)"
                    if envelope_enabled
                    else ""
                )
            ),
            "algorithm": (
                "hard_validity_plus_robust_target_envelope_v7" if envelope_enabled else
                "hard_validity_only_iqr_diagnostic_v1"
            ),
            "iqr_factor": args.iqr_factor,
            "iqr_role": "training_fitted_diagnostic_only",
            "iqr_invalidates_labels": False,
            "iqr_alters_model_input": False,
            "target_envelope_enabled": bool(envelope_enabled),
            "target_envelope_factor": args.target_envelope_factor,
            "target_envelope_floor": args.target_envelope_floor if envelope_enabled else None,
            "target_envelope_scope": list(TARGETS) if envelope_enabled else [],
            "target_envelope_fitted_on": "training split only" if envelope_enabled else TARGET_ENVELOPE_DISABLED,
            "target_envelope_statistic": "median (50% breakdown point) of absolute hard-valid training targets",
            "target_envelope_rules": target_envelope,
            "fitted_on": "training split only",
            "state_carry": "not_applicable_no_regime_state_machine",
            "retroactive_changes": False,
            "hard_invalid_recoverable": False,
            "train": train_cleaning_audit,
            "validation": validation_cleaning_audit,
            "test": test_cleaning_audit,
            "final_state": final_regime_state,
        },
        "zero_target_policy": args.zero_target_policy,
        "zero_target_interpretation": "physically valid and retained; counts are diagnostic only",
        "zero_target_counts": {
            "train": train_zero_counts,
            "validation": validation_zero_counts,
            "test": test_zero_counts,
        },
        "imputation": imputation,
        "label_validity": {
            "definition": "1 iff the raw target value is finite and passes all pre-registered quality rules; 0 otherwise",
            "quality_rules": (
                ["finite_numeric", "explicit_sentinel_abs_below_limit", "physical_bounds", "declared_source_quality_flags_if_present"]
                + (["robust_target_envelope"] if envelope_enabled else [])
            ),
            "iqr_affects_validity": False,
            "zero_affects_validity": False,
            "target_envelope_affects_validity": bool(envelope_enabled),
            "per_split_valid_counts": {name: {column: int(mask[column].sum()) for column in TARGETS} for name, mask in label_masks.items()},
            "per_split_invalid_counts": {name: {column: int((1.0 - mask[column]).sum()) for column in TARGETS} for name, mask in label_masks.items()},
            "evaluation_policy": "never impute validation/test labels; mask invalid target steps in loss and metrics",
        },
        "feature_screening": {
            "source": "assets/correlation_merged_data21-23/README.md",
            "threshold": args.selection_threshold,
            "score": "mean of absolute Pearson and Spearman correlations across the three targets",
            "selection": f"top {args.feature_count} ranked auxiliary features passing the training-only threshold after excluding targets and Combined mmBTU",
            "correlation_file": "train_correlations.csv",
        },
        "feature_availability": {
            "forecast_origin_rule": "every model input is observed at or before the forecast origin",
            "future_covariates_used": False,
            "GHG": "historical observations only; no future GHG value or target-derived Combined mmBTU is used",
            "timestamp_features": "deterministic calendar features known at the origin",
            "weather_features": "historical measured weather only; no future weather realization is used",
        },
        "zscore": {"fitted_on": "training split only", "target_source": "hard-valid training labels", "parameters": zstats},
        "sequence": {
            "policy": FORMAL_SEQUENCE_POLICY,
            "npz_written": False,
            "authorized_inputs": ["dataset_input.csv", "dataset_labels.csv", "label_valid_mask.csv"],
            "origin_source": "runner_generated_forecast_origins.csv_using_DataLoaderS_embargo",
            "horizons": [24, 48, 72, 96],
        },
    }
    artifact = {
        "artifact_version": 1,
        "protocol": metadata["protocol"],
        "input_sha256": metadata["input_sha256"],
        "split": metadata["split"],
        "targets": TARGETS,
        "candidate_features": candidates,
        "selected_features": selected_features,
        "model_columns": model_columns,
        "cleaning": {
            "rules": rules,
            "target_envelope": target_envelope,
            "target_envelope_enabled": bool(envelope_enabled),
            "target_envelope_factor": args.target_envelope_factor,
            "target_envelope_floor": args.target_envelope_floor if envelope_enabled else None,
            "fitted_on": "training split only",
            "algorithm": metadata["cleaning"]["algorithm"],
            "iqr_role": "training_fitted_diagnostic_only",
            "iqr_invalidates_labels": False,
            "state_carry": "not_applicable_no_regime_state_machine",
            "retroactive_changes": False,
            "hard_invalid_recoverable": False,
            "audit": {
                "train": train_cleaning_audit,
                "validation": validation_cleaning_audit,
                "test": test_cleaning_audit,
            },
        },
        "imputation": imputation,
        "feature_screening": metadata["feature_screening"],
        "feature_availability": metadata["feature_availability"],
        "source_quality_flags": metadata["source_quality_flags"],
        "sequence": metadata["sequence"],
        "generation": metadata["generation"],
        "zscore": metadata["zscore"],
        "label_validity": metadata["label_validity"],
    }
    artifact["artifact_sha256"] = canonical_hash(artifact)
    artifact_path = output_dir / "preprocessing_artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata["preprocessing_artifact"] = {
        "path": artifact_path.name,
        "sha256": sha256(artifact_path),
        "immutable": True,
    }

    report = [
        f"# Leakage-safe v{schema_version} model-input preprocessing record",
        f"- Input: `{input_path}` (SHA-256 `{metadata['input_sha256']}`)",
        f"- Model input: `{metadata['output_dataset']}` (SHA-256 `{metadata['output_dataset_sha256']}`)",
        f"- Time range: {metadata['time_start']} to {metadata['time_end']}; {n} aligned hourly rows.",
        f"- Chronological split: train {len(train)} / validation {len(validation)} / test {len(test)}.",
        f"- Targets: {', '.join(TARGETS)}; all three channels use kW.",
        f"- Selected model features ({len(selected_features)}): {', '.join(selected_features)}.",
        "- `Combined mmBTU` is excluded because it is derived from target loads.",
        f"- Feature ranking uses train-only absolute Pearson/Spearman correlation; threshold={args.selection_threshold:g}.",
        f"- Train-only {args.iqr_factor:g}*IQR bounds are diagnostics only and never remove or replace values.",
        "- Labels are invalid only for non-finite/parser failures, explicit sentinels, declared source-quality failures, or physical-bound violations.",
    ]
    if envelope_enabled:
        report.append(
            "- v7 adds a train-fitted robust target envelope: a target is invalid when "
            f"`abs(value) > max({args.target_envelope_floor:g}, {args.target_envelope_factor:g} * train median abs)`. "
            "The envelope uses the median (50% breakdown point) because an upper quantile is estimated "
            "from the contaminated sample and can be raised above the value it is meant to exclude."
        )
        for column, rule in target_envelope.items():
            report.append(
                f"  - `{column}`: median_abs={rule['median_abs']:.4f}, threshold={rule['threshold']:.4f}, "
                f"train max/median={rule['max_to_median_ratio']:.2f}"
            )
    report += [
        "- Physically plausible peaks, troughs, and zero target values are retained.",
        "- Only the model-input history is causally forward-filled; leading gaps use the training-fitted median.",
        "- Label validity means the raw target is finite and passes every pre-registered quality rule.",
        "- Formal experiments may read only `dataset_input.csv`, `dataset_labels.csv`, and `label_valid_mask.csv`; runners generate embargo origins.",
        "- No formal NPZ sequence artifact is written.",
        "- The directory was staged, fully validated, and atomically published; existing version directories are never overwritten.",
    ]
    record_path = output_dir / "PREPROCESSING_RECORD.md"
    record_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    metadata["output_files_sha256"] = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "preprocessing_metadata.json"
    }
    metadata["publication"] = {
        "immutable_destination": True,
        "staged_validation_complete": False,
        "atomic_rename_pending": True,
    }
    metadata_path = output_dir / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_staged_output(output_dir)
    metadata["publication"] = {
        "immutable_destination": True,
        "staged_validation_complete": True,
        "atomic_rename_pending": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "dataset_input": "dataset_input.csv",
        "model_columns": model_columns,
        "selected_features": selected_features,
        "rows": metadata["split"]["counts"],
        "sequence_policy": FORMAL_SEQUENCE_POLICY,
    }


def validate_staged_output(output_dir: Path) -> None:
    required = {
        "dataset_input.csv", "dataset_labels.csv", "label_valid_mask.csv",
        "preprocessing_artifact.json", "preprocessing_metadata.json", "PREPROCESSING_RECORD.md",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise ValueError(f"staged v6 output is incomplete: {missing}")
    forbidden = sorted(path.name for path in output_dir.iterdir() if path.suffix.lower() in {".npz", ".npy"})
    if forbidden:
        raise ValueError(f"formal v6 output contains forbidden sequence artifacts: {forbidden}")
    metadata = json.loads((output_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    schema_version = metadata.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"staged output schema version is unsupported: {schema_version!r}")
    if metadata.get("protocol") != SCHEMA_PROTOCOLS[schema_version]:
        raise ValueError("staged output protocol does not match its schema version")
    cleaning = metadata.get("cleaning", {})
    if schema_version == V7_SCHEMA_VERSION:
        if not cleaning.get("target_envelope_enabled"):
            raise ValueError("v7 staged output is missing the robust target envelope")
        if cleaning.get("target_envelope_fitted_on") != "training split only":
            raise ValueError("v7 target envelope is not marked training-only")
        for column in TARGETS:
            rule = cleaning.get("target_envelope_rules", {}).get(column)
            if not isinstance(rule, dict) or "threshold" not in rule:
                raise ValueError(f"v7 target envelope rule is missing for {column}")
    elif cleaning.get("target_envelope_enabled"):
        raise ValueError("v6 staged output may not enable the v7 target envelope")
    model_columns = list(metadata.get("model_columns", []))
    if len(model_columns) != 12 or model_columns[:3] != TARGETS:
        raise ValueError(f"invalid v6 model column contract: {model_columns}")
    inputs = pd.read_csv(output_dir / "dataset_input.csv")
    labels = pd.read_csv(output_dir / "dataset_labels.csv").to_numpy(dtype=float)
    mask = pd.read_csv(output_dir / "label_valid_mask.csv").to_numpy(dtype=float)
    if list(inputs.columns) != model_columns or inputs.shape[1] != 12:
        raise ValueError("dataset_input.csv header does not match metadata")
    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError("dataset_input.csv contains non-finite model inputs")
    if labels.shape != (len(inputs), 3) or mask.shape != labels.shape:
        raise ValueError("label sidecar dimensions do not match the model input")
    if not np.isin(mask, [0.0, 1.0]).all() or not np.array_equal(mask.astype(bool), np.isfinite(labels)):
        raise ValueError("label validity mask does not exactly match finite hard-valid labels")
    artifact = json.loads((output_dir / "preprocessing_artifact.json").read_text(encoding="utf-8"))
    recorded = artifact.pop("artifact_sha256", None)
    if recorded != canonical_hash(artifact):
        raise ValueError("preprocessing artifact content hash mismatch")
    for name, digest in metadata.get("output_files_sha256", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"staged output hash mismatch: {name}")


def main() -> None:
    args = parse_args()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"immutable v6 destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        summary = build_output(args, temporary)
        validate_staged_output(temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    print(json.dumps({"output_dir": str(destination), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
