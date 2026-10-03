import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Save_result import show_pred
from Save_result_multipredict import compute_report
from metrics import MAPE


LOAD_NAMES = ("electricity", "cooling", "heating")
HORIZONS = (24, 48, 72, 96)
MODEL_TO_RESULT = {
    "lstm": "supp-lstm",
    "gru": "supp-gru",
    "tcn": "supp-tcn",
}


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_csv(path):
    frame = pd.read_csv(path)
    raw = frame.to_numpy(dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"expected at least three columns in {path}, got shape {raw.shape}")
    return frame, raw


def load_label_sidecars(data_path, n_rows, evaluation_mode="validation-only"):
    base = Path(data_path).resolve().parent
    label_path = base / "dataset_labels.csv"
    mask_path = base / "label_valid_mask.csv"
    if not label_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            "versioned dataset_labels.csv and label_valid_mask.csv are required for deep baselines"
        )
    if evaluation_mode not in {"validation-only", "final-test"}:
        raise ValueError(f"unknown evaluation_mode {evaluation_mode!r}")
    train_end, valid_end = split_bounds(n_rows)
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


def fit_normalizer(raw, labels=None, label_mask=None):
    """Fit feature scaling on train rows, using observed labels for targets.

    Historical inputs remain the causally imputed ``raw`` values.  The first
    three normalization statistics are instead computed from observed target
    sidecar labels only, matching ``DataLoaderS`` and the main trainer.
    """
    train_end, _ = split_bounds(raw.shape[0])
    mean = raw[:train_end].mean(axis=0).astype(np.float64)
    std = raw[:train_end].std(axis=0).astype(np.float64)
    if labels is not None:
        observed = np.isfinite(labels[:train_end])
        if label_mask is not None:
            observed &= np.asarray(label_mask[:train_end], dtype=bool)
        for col in range(min(3, raw.shape[1])):
            values = labels[:train_end, col][observed[:, col]]
            if values.size == 0:
                raise ValueError(f"no observed training labels available for target column {col}")
            mean[col] = float(values.mean())
            std[col] = float(values.std())
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def make_xy(raw_norm, raw, indices, horizon, window, label_values=None, label_norm=None):
    x = np.empty((len(indices), window, raw_norm.shape[1]), dtype=np.float32)
    y_norm = np.empty((len(indices), horizon, 3), dtype=np.float32)
    y_true = np.empty((len(indices), horizon, 3), dtype=np.float32)
    for row, idx in enumerate(indices):
        origin = int(idx) - int(horizon)
        x[row] = raw_norm[origin - window + 1: origin + 1]
        y_norm_source = raw_norm if label_norm is None else label_norm
        y_norm[row] = y_norm_source[idx + 1 - horizon: idx + 1, :3]
        source = raw if label_values is None else label_values
        y_true[row] = source[idx + 1 - horizon: idx + 1, :3]
    return x, y_norm, y_true


class RNNForecaster(nn.Module):
    def __init__(self, cell, input_size, hidden_size, num_layers, dropout, horizon, out_nodes=3):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon * out_nodes)
        self.horizon = horizon
        self.out_nodes = out_nodes
        self.disable_cudnn = cell in {"gru", "lstm"}

    def forward(self, x):
        if self.disable_cudnn:
            with torch.backends.cudnn.flags(enabled=False):
                output, _ = self.rnn(x)
        else:
            output, _ = self.rnn(x)
        last = self.dropout(output[:, -1, :])
        return self.head(last).reshape(x.shape[0], self.horizon, self.out_nodes)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(y + residual)


class TCNForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, levels, kernel_size, dropout, horizon, out_nodes=3):
        super().__init__()
        blocks = []
        channels = [input_size] + [hidden_size] * levels
        for level in range(levels):
            blocks.append(
                TemporalBlock(
                    channels[level],
                    channels[level + 1],
                    kernel_size=kernel_size,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * out_nodes),
        )
        self.horizon = horizon
        self.out_nodes = out_nodes

    def forward(self, x):
        z = self.network(x.transpose(1, 2))
        last = z[:, :, -1]
        return self.head(last).reshape(x.shape[0], self.horizon, self.out_nodes)


def build_model(model_name, input_size, args, horizon):
    if model_name in {"lstm", "gru"}:
        return RNNForecaster(
            cell=model_name,
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            dropout=args.dropout,
            horizon=horizon,
        )
    if model_name == "tcn":
        return TCNForecaster(
            input_size=input_size,
            hidden_size=args.hidden_size,
            levels=args.tcn_levels,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
            horizon=horizon,
        )
    raise ValueError(f"unsupported model {model_name}")


def denorm_targets(y_norm, mean, std, device):
    mean_t = torch.as_tensor(mean[:3], dtype=y_norm.dtype, device=device).view(1, 1, 3)
    std_t = torch.as_tensor(std[:3], dtype=y_norm.dtype, device=device).view(1, 1, 3)
    return y_norm * std_t + mean_t


def mape_loss_original(y_pred_norm, y_true_norm, mean, std, device, label_mask=None):
    if label_mask is not None:
        valid = (
            label_mask.to(device=device, dtype=torch.bool)
            & torch.isfinite(y_true_norm)
            & torch.isfinite(y_pred_norm)
        )
        # Sanitize invalid elements before denormalization so NaNs cannot
        # poison the autograd graph; reduction remains element-wise masked.
        y_true_norm = torch.where(valid, y_true_norm, torch.zeros_like(y_true_norm))
        y_pred_norm = torch.where(valid, y_pred_norm, torch.zeros_like(y_pred_norm))
    else:
        valid = None
    y_pred = denorm_targets(y_pred_norm, mean, std, device)
    y_true = denorm_targets(y_true_norm, mean, std, device)
    loss = torch.abs(y_pred - y_true) / (torch.abs(y_true) + 1e-2)
    if valid is not None:
        if not torch.any(valid):
            return y_pred_norm.sum() * 0.0
        loss = loss.masked_select(valid)
    return loss.mean() * 100.0


@torch.no_grad()
def predict_original(model, loader, mean, std, device):
    model.eval()
    preds = []
    trues = []
    for batch in loader:
        xb, yb = batch[:2]
        xb = xb.to(device)
        yb = yb.to(device)
        pred_norm = model(xb)
        preds.append(denorm_targets(pred_norm, mean, std, device).cpu().numpy())
        trues.append(denorm_targets(yb, mean, std, device).cpu().numpy())
    return np.concatenate(trues, axis=0), np.concatenate(preds, axis=0)


def make_loader(x, y, batch_size, shuffle, label_mask=None):
    tensors = [torch.from_numpy(x), torch.from_numpy(y)]
    if label_mask is not None:
        tensors.append(torch.from_numpy(np.asarray(label_mask, dtype=np.bool_)))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def save_artifacts(result_dir, y_true, y_pred, horizon, metadata, split_name, label_mask=None):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = "val" if split_name == "validation" else "all"
    np.save(result_dir / f"{prefix}_y_true.npy", y_true.astype(np.float32))
    np.save(result_dir / f"{prefix}_predict_value.npy", y_pred.astype(np.float32))
    if label_mask is not None:
        np.save(result_dir / f"{prefix}_label_valid_mask.npy", np.asarray(label_mask, dtype=np.uint8))
    report = compute_report(y_true.astype(np.float32), y_pred.astype(np.float32), horizon=horizon, mask=label_mask)
    report["split"] = split_name
    report["test_materialized"] = metadata.get("evaluation_mode", split_name) == "final-test"
    report["test_accessed"] = metadata.get("evaluation_mode", split_name) == "final-test"
    metrics_name = "val_metrics_full.json" if split_name == "validation" else "metrics_full.json"
    with open(result_dir / metrics_name, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if split_name == "test":
        show_pred(y_true.astype(np.float32), y_pred.astype(np.float32), horizon, result_dir=str(result_dir), write_plots=False)
    with open(result_dir / "baseline_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    (result_dir / "run_summary.json").write_text(json.dumps({
        "status": "completed", "model": metadata.get("model"), "horizon": int(horizon),
        "seed": int(metadata.get("seed", 0)), "evaluation_mode": metadata.get("evaluation_mode", split_name),
        "evaluation_split": split_name,
        "test_materialized": metadata.get("evaluation_mode", split_name) == "final-test", "wall_seconds": metadata.get("wall_clock_seconds"),
        "test_accessed": metadata.get("evaluation_mode", split_name) == "final-test",
        "training_wall_seconds": metadata.get("wall_clock_seconds"),
        "training_wall_clock_seconds": metadata.get("wall_clock_seconds"),
        "model_parameters": metadata.get("parameter_count"),
        "trial_id": metadata.get("trial_id"), "config_hash": metadata.get("config_hash"),
        "selected_config": metadata.get("selected_config"),
        "training_time_definition": "wall-clock from process start through final checkpoint save; excludes HPO and locked-test evaluation",
    }, indent=2) + "\n", encoding="utf-8")
    return report


def train_one_model(model_name, horizon, raw, labels, label_mask, mean, std, args):
    run_start = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    raw_norm = (raw - mean) / std
    label_norm = None if labels is None else (labels - mean[:3]) / std[:3]
    train_idx = sample_indices(raw.shape[0], horizon, window=args.window, split="train", split_policy=args.split_policy)
    valid_idx = sample_indices(raw.shape[0], horizon, window=args.window, split="valid", split_policy=args.split_policy)
    _, valid_end = split_bounds(raw.shape[0])
    train_label_mask = make_xy(raw_norm, label_mask.astype(np.float32), train_idx, horizon, args.window)[2].astype(bool)
    if not np.any(train_label_mask):
        raise ValueError(f"no valid training labels for {model_name} {horizon}h")
    x_train, y_train, _ = make_xy(raw_norm, raw, train_idx, horizon, args.window, label_values=labels, label_norm=label_norm)
    x_valid, y_valid, valid_true = make_xy(raw_norm, raw, valid_idx, horizon, args.window, label_values=labels, label_norm=label_norm)
    valid_mask = make_xy(raw_norm, label_mask.astype(np.float32), valid_idx, horizon, args.window)[2].astype(bool)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True, label_mask=train_label_mask)
    valid_loader = make_loader(x_valid, y_valid, args.batch_size, shuffle=False, label_mask=valid_mask)

    model = build_model(model_name, raw.shape[1], args, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    result_dir = Path(args.result_dir) if args.result_dir else Path(args.result_root) / MODEL_TO_RESULT[model_name] / f"{horizon}-steps"
    model_dir = Path(args.model_dir) if args.model_dir else Path(args.model_root) / MODEL_TO_RESULT[model_name] / f"{horizon}-steps"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"
    rnn_backend = (
        "pytorch_cuda_fallback_cudnn_disabled"
        if model_name in {"gru", "lstm"} and device.type == "cuda"
        else "default"
    )
    config = dict(vars(args))
    config.update({"model": model_name, "horizon": horizon, "actual_lr_single": args.lr, "rnn_backend": rnn_backend,
                   "test_materialized": args.evaluation_mode == "final-test",
                   "test_accessed": args.evaluation_mode == "final-test"})
    with open(result_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    split_manifest = {
        "protocol": "new_dataset_3seed_v6_comparator",
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "input_dim": int(raw.shape[1]),
        "out_nodes": 3,
        "split_policy": args.split_policy,
        "evaluation_mode": args.evaluation_mode,
        "window_h": args.window,
        "horizon_h": horizon,
        "embargo_h": args.window + horizon if args.split_policy == "embargo" else 0,
        "sample_counts": {
            "train": len(train_idx),
            "validation": len(valid_idx),
            "test_available": raw.shape[0] - (split_bounds(raw.shape[0])[1] + (args.window + horizon if args.split_policy == "embargo" else 0)),
        },
        "target_end_index_ranges": {
            "train": [int(train_idx[0]), int(train_idx[-1])],
            "validation": [int(valid_idx[0]), int(valid_idx[-1])],
            "test": [int(valid_end + args.window + horizon if args.split_policy == "embargo" else valid_end), int(raw.shape[0] - 1)],
        },
        "test_materialized": args.evaluation_mode == "final-test",
        "test_accessed": args.evaluation_mode == "final-test",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None,
    }
    with open(result_dir / "split_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2)
        handle.write("\n")

    best_val = float("inf")
    best_epoch = None
    stale = 0
    history = []
    optimizer_steps_completed = 0
    budget_reached = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, batch_mask in train_loader:
            if optimizer_steps_completed >= args.max_optimizer_updates:
                budget_reached = True
                break
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mape_loss_original(pred, yb, mean, std, device, batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer_steps_completed += 1
            losses.append(float(loss.item()))

        if not losses:
            break

        _, valid_pred = predict_original(model, valid_loader, mean, std, device)
        val_mape = float(MAPE(valid_true[valid_mask], valid_pred[valid_mask]))
        if not np.isfinite(val_mape):
            raise ValueError("validation MAPE is not estimable under the active zero-label policy")
        history.append({"epoch": epoch, "train_mape_loss": float(np.mean(losses)), "valid_mape": val_mape})
        with open(result_dir / "validation_history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")
        print(
            f"{model_name} {horizon}h epoch {epoch:03d} | "
            f"train={np.mean(losses):.4f} | valid_mape={val_mape:.4f}",
            flush=True,
        )
        if val_mape < best_val:
            best_val = val_mape
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
        if optimizer_steps_completed >= args.max_optimizer_updates:
            budget_reached = True
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    if args.evaluation_mode == "validation-only":
        _, eval_pred = predict_original(model, valid_loader, mean, std, device)
        eval_true = valid_true
        eval_split = "validation"
        eval_indices = valid_idx
        eval_mask = valid_mask
    else:
        test_idx = sample_indices(raw.shape[0], horizon, window=args.window, split="test", split_policy=args.split_policy)
        x_test, y_test_norm, eval_true = make_xy(raw_norm, raw, test_idx, horizon, args.window, label_values=labels, label_norm=label_norm)
        test_loader = make_loader(x_test, y_test_norm, args.batch_size, shuffle=False)
        _, eval_pred = predict_original(model, test_loader, mean, std, device)
        eval_split = "test"
        eval_indices = test_idx
        eval_mask = make_xy(raw_norm, label_mask.astype(np.float32), test_idx, horizon, args.window)[2].astype(bool)
    metadata = {
        "model": model_name,
        "horizon": horizon,
        "split": "chronological 0.8/0.1/0.1 with explicit embargo",
        "evaluation_split": eval_split,
        "window": args.window,
        "normalization": "z-score fitted on training-only observed target labels for first three channels; causally imputed training inputs for covariates",
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "normalization_target_observed_counts": [int(np.isfinite(labels[:split_bounds(raw.shape[0])[0], col]).sum()) if labels is not None else int(split_bounds(raw.shape[0])[0]) for col in range(3)],
        "best_epoch": best_epoch,
        "best_valid_mape": best_val,
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "max_optimizer_updates": int(args.max_optimizer_updates),
        "optimizer_budget_reached": bool(budget_reached),
        "wall_clock_seconds": time.time() - run_start,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hardware_id": torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else str(device),
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "hyperparameters": {
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "dropout": args.dropout,
        "epochs": args.epochs,
            "max_optimizer_updates": args.max_optimizer_updates,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "tcn_levels": args.tcn_levels,
            "tcn_kernel_size": args.tcn_kernel_size,
            "seed": args.seed,
        },
        "training_history": history,
    }
    report = save_artifacts(result_dir, eval_true, eval_pred, horizon, metadata, eval_split, eval_mask)
    origins = eval_indices
    with (result_dir / "forecast_origins.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv
        metadata_path = Path(args.data).resolve().with_name("preprocessing_metadata.json")
        start_time = None
        if metadata_path.exists():
            try:
                metadata_json = json.loads(metadata_path.read_text(encoding="utf-8"))
                value = metadata_json.get("time_start")
                if value:
                    start_time = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                start_time = None
        fields = ["sample_index", "origin_index", "origin_timestamp", "origin_season", "target_start_index", "target_start_timestamp", "target_start_season", "target_end_index", "target_end_timestamp", "target_end_season", "horizon_h", "split"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_index, target_end in enumerate(np.asarray(origins, dtype=np.int64)):
            target_end = int(target_end)
            target_start = target_end - horizon + 1
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
                "horizon_h": horizon, "split": eval_split,
            })
    run_summary = {
        "trial_id": args.trial_id,
        "config_hash": args.config_hash,
        "status": "completed",
        "model": model_name,
        "horizon": horizon,
        "seed": args.seed,
        "evaluation_mode": args.evaluation_mode,
        "test_materialized": args.evaluation_mode == "final-test",
        "test_accessed": args.evaluation_mode == "final-test",
        "epochs_completed": len(history),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "max_optimizer_updates": int(args.max_optimizer_updates),
        "optimizer_budget_reached": bool(budget_reached),
        "best_epoch": best_epoch,
        "best_selection_metric": best_val,
        "selection_metric_name": "mape",
        "validation_or_test_metrics": report["overall"],
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "actual_learning_rates": {"single": args.lr, "cfc": None, "pmd": None, "cecm": None},
        "rnn_backend": rnn_backend,
        "wall_seconds": time.time() - run_start,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" and torch.cuda.is_available() else None,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" and torch.cuda.is_available() else None,
        "device": str(device),
    }
    with open(result_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
        handle.write("\n")
    by_load = {item["load"]: item for item in report["by_load"]}
    print(
        f"{model_name} {horizon}h | overall MAPE={report['overall']['mape']:.4f} "
        f"| ele={by_load['electricity']['mape']:.4f} "
        f"| cooling={by_load['cooling']['mape']:.4f} "
        f"| heating={by_load['heating']['mape']:.4f}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM/GRU/TCN energy-domain baselines.")
    parser.add_argument("--data", default="data/preprocessed_forecasting_v6/dataset_input.csv")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--model-root", default="model")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--models", nargs="+", default=["tcn"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--trial-id", default="")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--split-policy", choices=["legacy", "embargo"], default="embargo")
    parser.add_argument("--evaluation-mode", choices=["validation-only", "final-test"], default="validation-only")
    parser.add_argument("--window", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-optimizer-updates", type=int, default=12000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--tcn-levels", type=int, default=8)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    args = parser.parse_args()
    if args.max_optimizer_updates <= 0:
        parser.error("--max-optimizer-updates must be positive")
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
    fix_seed(args.seed)
    _, raw = load_raw_csv(args.data)
    labels, label_mask = load_label_sidecars(args.data, raw.shape[0], args.evaluation_mode)
    mean, std = fit_normalizer(raw, labels=labels, label_mask=label_mask)
    for model_name in args.models:
        for horizon in args.horizons:
            out_dir = Path(args.result_dir) if args.result_dir else Path(args.result_root) / MODEL_TO_RESULT[model_name] / f"{horizon}-steps"
            if out_dir.exists() and not args.overwrite:
                print(f"skip existing {out_dir}")
                continue
            train_one_model(model_name, horizon, raw, labels, label_mask, mean, std, args)


if __name__ == "__main__":
    main()
