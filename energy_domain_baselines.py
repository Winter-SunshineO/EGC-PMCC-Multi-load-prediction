import argparse
import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from Save_result import show_pred
from Save_result_multipredict import compute_report


LOAD_NAMES = ("electricity", "cooling", "heating")
HORIZONS = (24, 48, 72, 96)
MODEL_TO_RESULT = {
    "persistence": "supp-persistence",
    "daily-persistence": "supp-daily-persistence",
    "weekly-persistence": "supp-weekly-persistence",
    "lightgbm": "supp-lightgbm",
    "xgboost": "supp-xgboost",
}

LAG_HOURS = (1, 2, 3, 6, 12, 24, 48, 72, 96, 168)
ROLL_MEAN_HOURS = (3, 6, 12, 24, 48, 168)
ROLL_STD_HOURS = (6, 12, 24)
ROLL_MINMAX_HOURS = (24, 168)


def load_raw_csv(path):
    frame = pd.read_csv(path)
    raw = frame.to_numpy(dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"expected at least three columns in {path}, got shape {raw.shape}")
    return frame, raw


def load_label_sidecars(data_path, n_rows, evaluation_mode="validation-only"):
    """Load original target labels and their validity mask for versioned evaluation."""
    base = Path(data_path).resolve().parent
    label_path = base / "dataset_labels.csv"
    mask_path = base / "label_valid_mask.csv"
    if not label_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            "versioned dataset_labels.csv and label_valid_mask.csv are required for energy-domain baselines"
        )
    if evaluation_mode not in {"validation-only", "final-test"}:
        raise ValueError(f"unknown evaluation_mode {evaluation_mode!r}")
    _, valid_end = split_bounds(n_rows)
    limit = n_rows if evaluation_mode == "final-test" else valid_end
    labels = np.full((n_rows, 3), np.nan, dtype=np.float32)
    mask = np.zeros((n_rows, 3), dtype=bool)
    with label_path.open("r", encoding="utf-8-sig", newline="") as handle:
        loaded_labels = np.loadtxt(handle, delimiter=",", skiprows=1, max_rows=limit).astype(np.float32)
    with mask_path.open("r", encoding="utf-8-sig", newline="") as handle:
        loaded_mask = np.loadtxt(handle, delimiter=",", skiprows=1, max_rows=limit).astype(bool)
    if limit == 1:
        loaded_labels = np.asarray(loaded_labels).reshape(1, -1)
        loaded_mask = np.asarray(loaded_mask).reshape(1, -1)
    labels[:limit] = loaded_labels
    mask[:limit] = loaded_mask
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    if labels.shape != (n_rows, 3) or mask.shape != labels.shape or np.any(mask & ~np.isfinite(labels)):
        raise ValueError("versioned label sidecars are missing, malformed, or inconsistent")
    return labels, mask


def split_bounds(n, train_ratio=0.8, valid_ratio=0.1):
    train_end = int(train_ratio * n)
    valid_end = int((train_ratio + valid_ratio) * n)
    return train_end, valid_end


def sample_indices(n, horizon, window=168, split="test", split_policy="embargo"):
    train_end, valid_end = split_bounds(n)
    if split == "train":
        return np.arange(window + horizon - 1, train_end, dtype=np.int64)
    if split == "valid":
        start = train_end + window + horizon if split_policy == "embargo" else train_end
        return np.arange(start, valid_end, dtype=np.int64)
    if split == "test":
        start = valid_end + window + horizon if split_policy == "embargo" else valid_end
        return np.arange(start, n, dtype=np.int64)
    raise ValueError(f"unknown split {split}")


def make_targets(raw, indices, horizon):
    y = np.empty((len(indices), horizon, 3), dtype=np.float32)
    for row, idx in enumerate(indices):
        y[row] = raw[idx + 1 - horizon: idx + 1, :3]
    return y


def make_persistence_predictions(raw, indices, horizon, period=None):
    pred = np.empty((len(indices), horizon, 3), dtype=np.float32)
    steps = np.arange(horizon, dtype=np.int64)
    for row, idx in enumerate(indices):
        origin = int(idx) - int(horizon)
        if period is None:
            pred[row] = raw[origin, :3]
        else:
            source = origin - int(period) + 1 + (steps % int(period))
            pred[row] = raw[source, :3]
    return pred


def calendar_features(origin):
    hour = origin % 24
    week_hour = origin % 168
    day = (origin // 24) % 7
    return [
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * week_hour / 168.0),
        np.cos(2.0 * np.pi * week_hour / 168.0),
        np.sin(2.0 * np.pi * day / 7.0),
        np.cos(2.0 * np.pi * day / 7.0),
    ]


def build_feature_names(raw_columns):
    names = []
    for lag in LAG_HOURS:
        for load in LOAD_NAMES:
            names.append(f"{load}_lag_{lag}h")
    for window in ROLL_MEAN_HOURS:
        for load in LOAD_NAMES:
            names.append(f"{load}_roll_mean_{window}h")
    for window in ROLL_STD_HOURS:
        for load in LOAD_NAMES:
            names.append(f"{load}_roll_std_{window}h")
    for window in ROLL_MINMAX_HOURS:
        for stat in ("min", "max"):
            for load in LOAD_NAMES:
                names.append(f"{load}_roll_{stat}_{window}h")
    for prefix in ("current", "daily_lag", "weekly_lag"):
        for col in raw_columns[3:]:
            names.append(f"{prefix}_{col}")
    names.extend(["hour_sin", "hour_cos", "week_sin", "week_cos", "dow_sin", "dow_cos"])
    return names


def build_tree_features(raw, indices, raw_columns, horizon):
    features = np.empty((len(indices), len(build_feature_names(raw_columns))), dtype=np.float32)
    for row, idx in enumerate(indices):
        origin = int(idx) - int(horizon)  # most recent observed row before the first target step
        parts = []
        for lag in LAG_HOURS:
            parts.extend(raw[origin - lag + 1, :3])
        for window in ROLL_MEAN_HOURS:
            parts.extend(raw[origin - window + 1: origin + 1, :3].mean(axis=0))
        for window in ROLL_STD_HOURS:
            parts.extend(raw[origin - window + 1: origin + 1, :3].std(axis=0))
        for window in ROLL_MINMAX_HOURS:
            history = raw[origin - window + 1: origin + 1, :3]
            parts.extend(history.min(axis=0))
            parts.extend(history.max(axis=0))
        if raw.shape[1] > 3:
            parts.extend(raw[origin, 3:])
            parts.extend(raw[origin - 24 + 1, 3:])
            parts.extend(raw[origin - 168 + 1, 3:])
        parts.extend(calendar_features(origin))
        features[row] = np.asarray(parts, dtype=np.float32)
    return features


def result_dir_for(model_name, horizon, result_root):
    return Path(result_root) / MODEL_TO_RESULT[model_name] / f"{horizon}-steps"


def save_artifacts(result_dir, y_true, y_pred, horizon, metadata, split_name, label_mask=None):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = "val" if split_name == "validation" else "all"
    np.save(result_dir / f"{prefix}_y_true.npy", y_true)
    np.save(result_dir / f"{prefix}_predict_value.npy", y_pred)
    if label_mask is not None:
        np.save(result_dir / f"{prefix}_label_valid_mask.npy", np.asarray(label_mask, dtype=np.uint8))
    report = compute_report(y_true, y_pred, horizon=horizon, mask=label_mask)
    report["split"] = split_name
    report["test_materialized"] = metadata.get("evaluation_mode", split_name) == "final-test"
    report["test_accessed"] = metadata.get("evaluation_mode", split_name) == "final-test"
    metrics_name = "val_metrics_full.json" if split_name == "validation" else "metrics_full.json"
    with open(result_dir / metrics_name, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if split_name == "test":
        show_pred(y_true, y_pred, horizon, result_dir=str(result_dir), write_plots=False)
    with open(result_dir / "baseline_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    if "wall_clock_seconds" not in metadata:
        metadata["wall_clock_seconds"] = None
    run_summary = {
        "status": "completed", "model": metadata.get("model"), "horizon": int(horizon),
        "seed": int(metadata.get("seed", 0)), "evaluation_mode": metadata.get("evaluation_mode", split_name),
        "evaluation_split": split_name,
        "test_materialized": metadata.get("evaluation_mode", split_name) == "final-test", "wall_seconds": metadata.get("wall_clock_seconds"),
        "test_accessed": metadata.get("evaluation_mode", split_name) == "final-test",
        "training_wall_seconds": metadata.get("wall_clock_seconds"),
        "training_wall_clock_seconds": metadata.get("wall_clock_seconds"),
        "parameter_count": metadata.get("parameter_count"), "hardware_id": metadata.get("hardware_id"),
        "seed_invariant": bool(metadata.get("seed_invariant", False)),
        "n_seed_effective": int(metadata.get("n_seed_effective", 3 if not metadata.get("seed_invariant", False) else 1)),
        "trial_id": metadata.get("trial_id"),
        "config_hash": metadata.get("config_hash"),
        "selected_config": metadata.get("selected_config"),
        "training_time_definition": metadata.get("training_time_definition", "wall-clock from estimator start through validation prediction and artifact serialization"),
    }
    (result_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    return report


def write_protocol_artifacts(result_dir, n_rows, horizon, split_policy, evaluation_mode, args, eval_indices=None):
    """Write the same provenance contract as neural validation runs."""
    train_end, valid_end = split_bounds(n_rows)
    embargo = 168 + int(horizon) if split_policy == "embargo" else 0
    payload = {
        "protocol": "new_dataset_3seed_v6_comparator",
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "model": getattr(args, "model", "baseline"),
        "seed": int(args.seed),
        "input_dim": 12,
        "out_nodes": 3,
        "window_h": 168,
        "horizon_h": int(horizon),
        "embargo_h": int(embargo),
        "split_policy": split_policy,
        "evaluation_mode": evaluation_mode,
        "checkpoint_policy": "not_applicable",
        "test_materialized": evaluation_mode == "final-test",
        "test_accessed": evaluation_mode == "final-test",
        "sample_counts": {
            "train": max(0, train_end - (168 + horizon - 1)),
            "validation": max(0, valid_end - (train_end + embargo)),
            "test_available": max(0, n_rows - (valid_end + embargo)),
        },
        "target_end_index_ranges": {
            "train": [168 + horizon - 1, train_end - 1],
            "validation": [train_end + embargo, valid_end - 1],
            "test": [valid_end + embargo, n_rows - 1],
        },
    }
    result_dir = Path(result_dir)
    result_dir.joinpath("split_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result_dir.joinpath("config.json").write_text(json.dumps({**vars(args), **payload}, indent=2, default=str) + "\n", encoding="utf-8")
    if eval_indices is not None:
        import csv
        metadata_path = Path(args.data).resolve().with_name("preprocessing_metadata.json")
        start_time = None
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                value = metadata.get("time_start")
                if value:
                    start_time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                start_time = None
        with result_dir.joinpath("forecast_origins.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["sample_index", "origin_index", "origin_timestamp", "origin_season", "target_start_index", "target_start_timestamp", "target_start_season", "target_end_index", "target_end_timestamp", "target_end_season", "horizon_h", "split"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample_index, target_end in enumerate(np.asarray(eval_indices, dtype=np.int64)):
                target_end = int(target_end)
                target_start = target_end - int(horizon) + 1
                origin = target_start - 1
                def stamp(index):
                    return (start_time + timedelta(hours=int(index))).isoformat() if start_time is not None else ""
                origin_timestamp = stamp(origin)
                target_start_timestamp = stamp(target_start)
                target_end_timestamp = stamp(target_end)
                def season(timestamp):
                    if not timestamp:
                        return ""
                    month = datetime.fromisoformat(timestamp).month
                    return "DJF" if month in (12, 1, 2) else "MAM" if month in (3, 4, 5) else "JJA" if month in (6, 7, 8) else "SON"
                writer.writerow({
                    "sample_index": sample_index, "origin_index": origin,
                    "origin_timestamp": origin_timestamp, "origin_season": season(origin_timestamp),
                    "target_start_index": target_start, "target_start_timestamp": target_start_timestamp,
                    "target_start_season": season(target_start_timestamp),
                    "target_end_index": target_end, "target_end_timestamp": target_end_timestamp,
                    "target_end_season": season(target_end_timestamp),
                    "horizon_h": int(horizon),
                    "split": "test" if evaluation_mode == "final-test" else "validation",
                })


def print_short_report(model_name, horizon, report):
    by_load = {item["load"]: item for item in report["by_load"]}
    print(
        f"{model_name} {horizon}h | overall MAPE={report['overall']['mape']:.4f} "
        f"| ele={by_load['electricity']['mape']:.4f} "
        f"| cooling={by_load['cooling']['mape']:.4f} "
        f"| heating={by_load['heating']['mape']:.4f}",
        flush=True,
    )


def run_persistence_models(raw, labels, label_mask, horizons, models, result_root, overwrite, split_policy, evaluation_mode, args):
    for horizon in horizons:
        split = "test" if evaluation_mode == "final-test" else "valid"
        eval_idx = sample_indices(raw.shape[0], horizon, split=split, split_policy=split_policy)
        y_true = make_targets(labels if labels is not None else raw, eval_idx, horizon)
        y_mask = make_targets(label_mask.astype(np.float32), eval_idx, horizon).astype(bool)
        for model_name, period in [
            ("persistence", None),
            ("daily-persistence", 24),
            ("weekly-persistence", 168),
        ]:
            if model_name not in models:
                continue
            run_start = time.perf_counter()
            out_dir = Path(args.result_dir) if args.result_dir else result_dir_for(model_name, horizon, result_root)
            if out_dir.exists() and not overwrite:
                print(f"skip existing {out_dir}")
                continue
            y_pred = make_persistence_predictions(raw, eval_idx, horizon, period=period)
            metadata = {
                "model": model_name,
                "horizon": horizon,
                "split": "chronological 0.8/0.1/0.1 with explicit embargo",
                "window": 168,
                "split_policy": split_policy,
                "evaluation_mode": evaluation_mode,
                "evaluation_split": "test" if split == "test" else "validation",
                "seed": args.seed,
                "period": period,
                "seed_invariant": True,
                "n_seed_effective": 1,
                "leakage_guard": "seasonal persistence repeats the last fully observed period for all forecast steps",
                "wall_clock_seconds": time.perf_counter() - run_start,
                "trial_id": getattr(args, "trial_id", "not_applicable"),
                "config_hash": getattr(args, "config_hash", ""),
                "selected_config": {"lag": period},
                "training_time_definition": "wall-clock from baseline prediction start through validation prediction and artifact serialization",
                "parameter_count": 0,
                "hardware_id": "cpu",
            }
            report = save_artifacts(out_dir, y_true, y_pred, horizon, metadata, "test" if split == "test" else "validation", y_mask)
            write_protocol_artifacts(out_dir, raw.shape[0], horizon, split_policy, evaluation_mode, args, eval_idx)
            print_short_report(model_name, horizon, report)


def make_tree_estimator(model_name, args):
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            from sklearn.multioutput import MultiOutputRegressor

            base = LGBMRegressor(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                min_child_samples=args.min_child_samples,
                random_state=args.seed,
                n_jobs=1,
                verbosity=-1,
            )
            return MultiOutputRegressor(base, n_jobs=args.n_jobs), "lightgbm.LGBMRegressor"
        except ImportError:
            if not args.allow_sklearn_fallback:
                raise
            return make_sklearn_hist_fallback(args), "sklearn.HistGradientBoostingRegressor fallback for LightGBM"

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor

            base = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                min_child_weight=args.min_child_weight,
                tree_method="hist",
                random_state=args.seed,
                n_jobs=args.n_jobs,
                early_stopping_rounds=50,
            )
            return MultiOutputRegressor(base, n_jobs=args.n_jobs), "xgboost.XGBRegressor"
        except ImportError:
            if not args.allow_sklearn_fallback:
                raise
            return make_sklearn_hist_fallback(args), "sklearn.HistGradientBoostingRegressor fallback for XGBoost"

    raise ValueError(f"unsupported tree model {model_name}")


def fit_tree_with_validation(estimator, model_name, x_train, y_train, x_valid, y_valid, valid_mask, train_mask=None):
    """Fit one estimator per flattened output with a validation eval_set.

    ``MultiOutputRegressor.fit`` cannot reliably route a two-dimensional
    validation target to LightGBM/XGBoost across supported sklearn versions,
    so the individual estimators are fitted explicitly and stored back on the
    wrapper.  Missing validation labels are excluded per output channel.
    """
    from sklearn.base import clone

    if not hasattr(estimator, "estimator"):
        raise TypeError("tree estimator must expose a per-output estimator for masked fitting")
    fitted = []
    best_iterations = []
    used_eval_set = False
    valid_mask = np.asarray(valid_mask, dtype=bool)
    train_mask = np.ones_like(y_train, dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    for column in range(y_train.shape[1]):
        base = clone(estimator.estimator)
        train_keep = train_mask[:, column] & np.isfinite(y_train[:, column])
        if not np.any(train_keep):
            raise ValueError(f"no valid training labels for tree output column {column}")
        x_fit = x_train[train_keep]
        y_fit = y_train[train_keep, column]
        x_val = x_valid
        y_val = y_valid[:, column]
        keep = valid_mask[:, column] & np.isfinite(y_val)
        fit_kwargs = {}
        if np.any(keep):
            x_val = x_valid[keep]
            y_val = y_val[keep]
            if model_name == "lightgbm" and base.__class__.__module__.startswith("lightgbm"):
                from lightgbm import early_stopping
                fit_kwargs = {"eval_set": [(x_val, y_val)], "callbacks": [early_stopping(50, verbose=False)]}
            elif model_name == "xgboost" and base.__class__.__module__.startswith("xgboost"):
                fit_kwargs = {"eval_set": [(x_val, y_val)], "verbose": False}
            used_eval_set = True
        try:
            base.fit(x_fit, y_fit, **fit_kwargs)
        except TypeError:
            # Older XGBoost versions expose early stopping as a fit keyword.
            if model_name == "xgboost" and fit_kwargs:
                fit_kwargs = {"eval_set": fit_kwargs["eval_set"], "verbose": False, "early_stopping_rounds": 50}
                base.fit(x_fit, y_fit, **fit_kwargs)
            else:
                raise
        fitted.append(base)
        best_iterations.append(getattr(base, "best_iteration_", getattr(base, "best_iteration", None)))
    estimator.estimators_ = fitted
    return best_iterations, used_eval_set


def make_sklearn_hist_fallback(args):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    base = HistGradientBoostingRegressor(
        max_iter=args.n_estimators,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.num_leaves,
        l2_regularization=0.0,
        random_state=args.seed,
    )
    return MultiOutputRegressor(base, n_jobs=args.n_jobs)


def run_tree_models(frame, raw, labels, label_mask, horizons, models, result_root, overwrite, args):
    raw_columns = list(frame.columns)
    feature_names = build_feature_names(raw_columns)
    for horizon in horizons:
        train_idx = sample_indices(raw.shape[0], horizon, split="train")
        eval_split = "test" if args.evaluation_mode == "final-test" else "valid"
        validation_idx = sample_indices(raw.shape[0], horizon, split="valid", split_policy=args.split_policy)
        eval_idx = sample_indices(raw.shape[0], horizon, split=eval_split, split_policy=args.split_policy)
        x_train = build_tree_features(raw, train_idx, raw_columns, horizon)
        y_train_full = make_targets(labels if labels is not None else raw, train_idx, horizon)
        y_train_mask = make_targets(label_mask.astype(np.float32), train_idx, horizon).astype(bool)
        y_train = y_train_full.reshape(len(train_idx), -1)
        y_train_mask_flat = y_train_mask.reshape(len(train_idx), -1)
        x_valid = build_tree_features(raw, validation_idx, raw_columns, horizon)
        y_valid = make_targets(labels if labels is not None else raw, validation_idx, horizon)
        y_valid_mask = make_targets(label_mask.astype(np.float32), validation_idx, horizon).astype(bool)
        y_valid_flat = y_valid.reshape(len(validation_idx), -1)
        y_valid_mask_flat = y_valid_mask.reshape(len(validation_idx), -1)
        x_eval = build_tree_features(raw, eval_idx, raw_columns, horizon)
        y_true = make_targets(labels if labels is not None else raw, eval_idx, horizon)
        y_eval_mask = make_targets(label_mask.astype(np.float32), eval_idx, horizon).astype(bool)
        for model_name in ["lightgbm", "xgboost"]:
            if model_name not in models:
                continue
            out_dir = Path(args.result_dir) if args.result_dir else result_dir_for(model_name, horizon, result_root)
            model_dir = Path(args.model_dir) if args.model_dir else Path(result_root) / "model" / model_name / f"{horizon}h"
            if out_dir.exists() and not overwrite:
                print(f"skip existing {out_dir}")
                continue
            estimator, implementation = make_tree_estimator(model_name, args)
            run_start = time.perf_counter()
            print(
                f"training {model_name} {horizon}h with {implementation} "
                f"| X={x_train.shape} | Y={y_train.shape}",
                flush=True,
            )
            best_iterations, used_eval_set = fit_tree_with_validation(
                estimator, model_name, x_train, y_train, x_valid, y_valid_flat, y_valid_mask_flat, y_train_mask_flat
            )
            model_dir.mkdir(parents=True, exist_ok=True)
            with (model_dir / "best.pt").open("wb") as handle:
                pickle.dump(estimator, handle, protocol=pickle.HIGHEST_PROTOCOL)
            y_pred = estimator.predict(x_eval).reshape(len(eval_idx), horizon, 3).astype(np.float32)
            metadata = {
                "model": model_name,
                "implementation": implementation,
                "horizon": horizon,
                "split": "chronological 0.8/0.1/0.1 with explicit embargo",
                "window": 168,
                "split_policy": args.split_policy,
                "evaluation_mode": args.evaluation_mode,
                "evaluation_split": "test" if eval_split == "test" else "validation",
                "seed": args.seed,
                "wall_clock_seconds": time.perf_counter() - run_start,
                "parameter_count": 0,
                "hardware_id": "cpu",
                "model_checkpoint": str(model_dir / "best.pt"),
                "feature_groups": {
                    "lagged_loads_h": list(LAG_HOURS),
                    "rolling_mean_h": list(ROLL_MEAN_HOURS),
                    "rolling_std_h": list(ROLL_STD_HOURS),
                    "rolling_min_max_h": list(ROLL_MINMAX_HOURS),
                    "past_covariates": ["current", "daily_lag", "weekly_lag"],
                    "calendar_from_index": ["hour", "day_of_week", "week_hour"],
                },
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "leakage_guard": "features use only rows at or before the most recent observed origin",
                "hyperparameters": {
                    "n_estimators": args.n_estimators,
                    "learning_rate": args.learning_rate,
                    "num_leaves": args.num_leaves,
                    "max_depth": args.max_depth,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "seed": args.seed,
                },
                "trial_id": getattr(args, "trial_id", "selected"),
                "config_hash": getattr(args, "config_hash", ""),
                "selected_config": {
                    "n_estimators": args.n_estimators,
                    "learning_rate": args.learning_rate,
                    "num_leaves": args.num_leaves,
                    "max_depth": args.max_depth,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "min_child_samples": getattr(args, "min_child_samples", None),
                    "min_child_weight": getattr(args, "min_child_weight", None),
                },
                "training_time_definition": "wall-clock from estimator.fit start through validation prediction and artifact serialization",
                "early_stopping_rounds": 50,
                "validation_eval_set": bool(used_eval_set),
                "early_stopping_split": "validation",
                "early_stopping_sample_count": int(len(validation_idx)),
                "best_iterations": best_iterations,
            }
            report = save_artifacts(out_dir, y_true, y_pred, horizon, metadata, "test" if eval_split == "test" else "validation", y_eval_mask)
            write_protocol_artifacts(out_dir, raw.shape[0], horizon, args.split_policy, args.evaluation_mode, args, eval_idx)
            print_short_report(model_name, horizon, report)


def parse_args():
    parser = argparse.ArgumentParser(description="Energy-domain persistence and boosted-tree baselines.")
    parser.add_argument("--data", default="data/preprocessed_forecasting_v6/dataset_input.csv")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--result-dir", default=None, help="isolated output directory for one model x horizon run")
    parser.add_argument("--model-dir", default=None, help="isolated checkpoint directory for one model x horizon run")
    parser.add_argument("--models", nargs="+", default=["persistence", "daily-persistence", "weekly-persistence"])
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-sklearn-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--trial-id", default="")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--split-policy", choices=["legacy", "embargo"], default="embargo")
    parser.add_argument("--evaluation-mode", choices=["validation-only", "final-test"], default="validation-only")
    args = parser.parse_args()
    unknown = sorted(set(args.models) - set(MODEL_TO_RESULT))
    if unknown:
        parser.error(f"unknown models: {unknown}")
    bad_horizons = sorted(set(args.horizons) - set(HORIZONS))
    if bad_horizons:
        parser.error(f"horizons must be in {HORIZONS}, got {bad_horizons}")
    if (args.result_dir or args.model_dir) and (len(args.models) != 1 or len(args.horizons) != 1):
        parser.error("--result-dir/--model-dir require exactly one model and one horizon")
    return args


def main():
    args = parse_args()
    frame, raw = load_raw_csv(args.data)
    labels, label_mask = load_label_sidecars(args.data, raw.shape[0], args.evaluation_mode)
    models = set(args.models)
    run_persistence_models(raw, labels, label_mask, args.horizons, models, args.result_root, args.overwrite, args.split_policy, args.evaluation_mode, args)
    tree_models = models & {"lightgbm", "xgboost"}
    if tree_models:
        run_tree_models(frame, raw, labels, label_mask, args.horizons, models, args.result_root, args.overwrite, args)


if __name__ == "__main__":
    main()
