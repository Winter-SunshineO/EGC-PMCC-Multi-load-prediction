"""Original inference and timing kernels with portable data paths."""
from contextlib import contextmanager
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import train
import deep_energy_baselines as deep
import energy_domain_baselines as energy
from release_common import DATA, read_json, require, sha256, canonical_hash
recipe=SimpleNamespace(train=train)
base=SimpleNamespace(DATA=DATA, POINT_STAGES=("p0","p2a","p2b"))
main=SimpleNamespace(DETERMINISTIC=("persistence","daily_persistence","weekly_persistence"),recipe=recipe)
PROFILE="embedded_parent_requires_grad_match_under_no_grad_v1"
def check(payload, field):
    require(payload[field]==canonical_hash({k:v for k,v in payload.items() if k!=field}), "Scaler hash mismatch")


def restore_scaler(data, config, result):
    if config.get("point_training_recipe") != "a3":
        return
    scaler = read_json(result / "scaler.json")
    check(scaler, "scaler_sha256")
    if (scaler["scaler_sha256"] != config["target_scaler_sha256"]
            or scaler.get("fit_split") != "train_only" or scaler.get("fit_rows") != 21024):
        raise ValueError("A3 inference scaler differs from the frozen training scaler")
    recipe.train.apply_target_robust_scaler(data, scaler)

def neural_forward(entry, device="cuda:0", split="test"):
    if split not in ("test", "validation"):
        raise ValueError("unsupported inference split")
    horizon = entry["horizon"]
    result = Path(entry["unit"]) / "result"
    cfg = read_json(result / "config.json")
    data = recipe.train.DataLoaderS(str(base.DATA), .8, .1, torch.device(device),
                                   horizon, 168, cfg["normalize"],
                                   split_policy="embargo", materialize_test=split == "test")
    restore_scaler(data, cfg, result)
    inputs, targets = data.test if split == "test" else data.valid
    mask = data.test_label_mask if split == "test" else data.valid_label_mask
    model = recipe.train.load_checkpoint_cpu(entry["checkpoint"]).to(device).eval()
    values = recipe.train.plow(data, inputs, targets[:, :, :3], model, 256, horizon, 3,
                               label_mask=mask, return_mask=True, return_cecm_correction=True)
    y, pred, q, mask, correction = [v.detach().cpu().numpy() if v is not None else None for v in values]
    return {"y": y, "pred": pred, "mask": mask, "q": q,
            "levels": [float(v) for v in model.quantiles] if q is not None else None,
            "correction": correction, "first_target_end": int(data.split_metadata["target_end_index_ranges"][split][0])}

def infer(entry, device="cuda:0", split="test"):
    """One logical pass, using the existing domain/model implementations."""
    if split not in ("test", "validation"):
        raise ValueError("unsupported inference split")
    method, horizon = entry["method"], entry["horizon"]
    result = Path(entry["unit"]) / "result"
    q, correction, levels = None, None, None
    if method in (*base.POINT_STAGES, "p4", "itransformer"):
        return neural_forward(entry, device, split)
    else:
        loader = deep if method in ("gru", "lstm", "tcn") else energy
        frame, raw = loader.load_raw_csv(base.DATA)
        labels, valid = loader.load_label_sidecars(
            base.DATA, len(raw), evaluation_mode="final-test" if split == "test" else "validation-only")
        indices = energy.sample_indices(len(raw), horizon, split="test" if split == "test" else "valid",
                                        split_policy="embargo")
        y = energy.make_targets(labels, indices, horizon)
        mask = energy.make_targets(valid.astype(np.float32), indices, horizon).astype(bool)
        first = int(indices[0])
        if method in main.DETERMINISTIC:
            period = {"persistence": None, "daily_persistence": 24, "weekly_persistence": 168}[method]
            pred = energy.make_persistence_predictions(raw, indices, horizon, period=period)
        elif method == "lightgbm":
            with Path(entry["checkpoint"]).open("rb") as handle:
                model = pickle.load(handle)
            x = energy.build_tree_features(raw, indices, list(frame.columns), horizon)
            pred = model.predict(x).reshape(len(indices), horizon, 3)
        else:
            cfg = read_json(result / "config.json")
            mean, std = deep.fit_normalizer(raw, labels=labels, label_mask=valid)
            x, _, _ = deep.make_xy((raw - mean) / std, raw, indices, horizon, 168, label_values=labels)
            model = deep.build_model(method, raw.shape[1], SimpleNamespace(**cfg), horizon).to(device)
            model.load_state_dict(torch.load(entry["checkpoint"], map_location=device, weights_only=False))
            model.eval()
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(x), 256):
                    normalized = model(torch.from_numpy(x[start:start + 256]).to(device))
                    predictions.append(deep.denorm_targets(normalized, mean, std, device).cpu().numpy())
            pred = np.concatenate(predictions)
    return {"y": y, "pred": pred, "mask": mask, "q": q, "levels": levels,
            "correction": correction, "first_target_end": int(first)}

def verify_parent_state(model, parent):
    if not getattr(model, "freeze_point", False):
        raise ValueError("P4 frozen-parent contract required")
    embedded = model.point_model
    a, b = embedded.state_dict(), parent.state_dict()
    if (type(embedded) is not type(parent) or a.keys() != b.keys()
            or any(not torch.equal(a[k].cpu(), b[k].cpu()) for k in a)):
        raise ValueError("embedded parent is not exactly the frozen parent")
    children = dict(embedded.named_modules())
    source = dict(parent.named_modules())
    if children.keys() != source.keys() or any(type(children[k]) is not type(source[k]) for k in children):
        raise ValueError("parent module structure differs")

@contextmanager
def matching_execution_flags(model, parent):
    """Match dispatch metadata only; no optimizer, backward, or parameter edit."""
    verify_parent_state(model, parent)
    embedded = dict(model.point_model.named_parameters())
    source = dict(parent.named_parameters())
    if embedded.keys() != source.keys():
        raise ValueError("parent parameter names differ")
    original = {k: v.requires_grad for k, v in embedded.items()}
    changed = {k: {"before": original[k], "during_inference": source[k].requires_grad}
               for k in embedded if original[k] != source[k].requires_grad}
    try:
        for name, value in embedded.items():
            value.requires_grad_(source[name].requires_grad)
        with torch.no_grad():
            yield {"profile": PROFILE, "changed_execution_flags": changed,
                   "freeze_point": True, "gradient_recording": False,
                   "weights_modified": False, "checkpoint_modified": False}
    finally:
        for name, value in embedded.items():
            value.requires_grad_(original[name])

def latency_input(entry):
    """Construct the registered validation batch before any timed region."""
    import deep_energy_baselines as deep
    import energy_domain_baselines as energy
    from training_protocols import apply_target_robust_scaler
    from util import DataLoaderS
    method, h = entry["method"], int(entry["horizon"])
    result = Path(entry["unit"]) / "result"
    checkpoint = entry.get("checkpoint")
    if checkpoint:
        require(sha256(checkpoint) == entry["checkpoint_sha256"], "latency checkpoint hash changed")
    cfg = read_json(result / "config.json")
    matched = entry.get("family") == "S2"
    if matched or method in (*base.POINT_STAGES, "p4", "itransformer"):
        device = torch.device(os.environ.get("EGC_DEVICE", "cuda:0"))
        data = DataLoaderS(str(base.DATA), .8, .1, device, h, 168, cfg.get("normalize", 2),
                           split_policy="embargo", materialize_test=False)
        if matched:
            from reviewer_cells import MatchedRecurrentForecaster
            apply_target_robust_scaler(data, read_json(result / "scaler.json"))
            model = MatchedRecurrentForecaster(
                cfg["variant"], cfg["hidden_size"], h, cfg["period_set"],
                common_hidden=cfg["common_hidden"], dropout=cfg["dropout"]).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        else:
            restore_scaler(data, cfg, result)
            model = main.recipe.train.load_checkpoint_cpu(checkpoint).to(device)
        require(data.test is None and data.split_metadata["label_sidecars"]["loaded_through_row_exclusive"] == 23652,
                "latency loader crossed validation boundary")
        x = data.valid[0][:256].clone().to(device=device, dtype=torch.float32)
        if not matched:
            x = x.unsqueeze(1).transpose(2, 3)
        del data
        model.eval()
        return lambda: model(x), device, tuple(x.shape), 21024+168
    loader = deep if method in ("gru", "lstm", "tcn") else energy
    frame, raw = loader.load_raw_csv(base.DATA)
    indices = energy.sample_indices(len(raw), h, split="valid", split_policy="embargo")[:256]
    require(len(indices) == 256 and int(indices[0])-h == 21024+168, "latency origin changed")
    if method in main.DETERMINISTIC:
        raw = raw[:23652].astype(np.float32)
        period = {"persistence": None, "daily_persistence": 24, "weekly_persistence": 168}[method]
        return lambda: energy.make_persistence_predictions(raw, indices, h, period=period), torch.device("cpu"), (256, 168, 3), int(indices[0])-h
    if method == "lightgbm":
        x = energy.build_tree_features(raw, indices, list(frame.columns), h).astype(np.float32)
        with Path(checkpoint).open("rb") as handle:
            estimator = pickle.load(handle)
        # Keep output estimators sequential; constrain each tree's native threads.
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        for child in getattr(estimator, "estimators_", [estimator]):
            if hasattr(child, "set_params") and "n_jobs" in child.get_params():
                child.set_params(n_jobs=3)
        return lambda: estimator.predict(x), torch.device("cpu"), tuple(x.shape), int(indices[0])-h
    require(method in ("gru", "lstm", "tcn"), "unknown latency model")
    labels, mask = deep.load_label_sidecars(base.DATA, len(raw), evaluation_mode="validation-only")
    mean, std = deep.fit_normalizer(raw, labels=labels, label_mask=mask)
    normalized = (raw-mean)/std
    x = np.stack([normalized[int(i)-h-167:int(i)-h+1] for i in indices]).astype(np.float32)
    device = torch.device(os.environ.get("EGC_DEVICE", "cuda:0"))
    x = torch.from_numpy(x).to(device)
    model = deep.build_model(method, raw.shape[1], SimpleNamespace(**cfg), h).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    return lambda: model(x), device, tuple(x.shape), int(indices[0])-h

def measure_blocks(call, device, warmups=10, blocks=30, repeats=20):
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    values = []
    with torch.inference_mode(), threadpool_limits(limits=3):
        for _ in range(warmups):
            output = call()
        if torch.is_tensor(output):
            require(torch.isfinite(output).all().item(), "nonfinite latency warmup")
        elif isinstance(output, np.ndarray):
            require(np.isfinite(output).all(), "nonfinite latency warmup")
        sync()
        for block in range(blocks):
            sync()
            start = time.perf_counter_ns()
            for _ in range(repeats):
                output = call()
            sync()
            elapsed = (time.perf_counter_ns()-start)/1e6
            values.append(dict(block=block, repeats=repeats, block_ms=elapsed,
                               batch_latency_ms=elapsed/repeats))
    return values

def seasons(timestamps):
    months = pd.to_datetime(timestamps, errors="raise").dt.month
    require(months.notna().all(), "missing origin timestamp")
    return months.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                       6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})

def parameter_rows(snapshot):
    rows = []
    for key in ("cecm_gate", "cecm_gate_logit"):
        value = float(snapshot[key])
        require(np.isfinite(value), "nonfinite CECM parameter")
        rows.append(dict(parameter=key, row=-1, column=-1, value=value))
    if snapshot["is_static_matrix"]:
        keys = ("delta_matrix", "effective_correction_matrix")
    else:
        require(snapshot["delta_matrix"] is None and snapshot["effective_correction_matrix"] is None,
                "conditional CECM must not be represented as a static matrix")
        keys = ("delta_intercept_matrix", "delta_slope_matrix",
                "delta_season_sin_matrix", "delta_season_cos_matrix")
        require(snapshot["calendar"]["ready"], "season calendar not initialized")
    for key in keys:
        matrix = np.asarray(snapshot[key], dtype=np.float64)
        require(matrix.shape == (3, 3) and np.isfinite(matrix).all(), "invalid CECM matrix")
        for i in range(3):
            for j in range(3):
                rows.append(dict(parameter=key, row=i, column=j, value=float(matrix[i, j])))
                if not snapshot["is_static_matrix"]:
                    rows.append(dict(parameter=f"gated_{key}", row=i, column=j,
                                     value=float(matrix[i, j]*snapshot["cecm_gate"])))
    return rows
