import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn

from Save_result import show_pred
from cfc_pci_heads import CfcWithPMD, CfcWithPMDCECM
from linear_baselines import DLinear, NLinear
from metrics import MAE, MAPE, RMSE, correlation, metric_by_horizon, metric_by_load, metric_by_load_and_horizon, metric_overall
from quantile_heads import CfcPMDCECMQuantile
from sota_baselines import ITransformerBaseline, PatchTSTBaseline, TimeMixerBaseline, TimesNetBaseline, SMambaBaseline
from torch_cfc import Cfc
from trainer import MultiGroupPlateauOptim, Optim
from training_protocols import (
    A3_RECIPE,
    LEGACY_RECIPE,
    apply_target_robust_scaler,
    bounded_percentage_loss_physical,
    build_a3_protocol_manifest,
    fit_target_bounded_tau,
    fit_target_robust_scaler,
)
from util import DataLoaderS, mape_loss

warnings.filterwarnings("ignore")


class LegacyCheckpointPolicyWarning(UserWarning):
    pass


warnings.filterwarnings("always", category=LegacyCheckpointPolicyWarning)

HORIZON_PRESETS = {
    24: {"epochs": 39, "lr": 0.008, "patience": 2, "lr_d": 0.5},
    48: {"epochs": 39, "lr": 0.008, "patience": 1, "lr_d": 0.45},
    72: {"epochs": 39, "lr": 0.008, "patience": 1, "lr_d": 0.45},
    96: {"epochs": 39, "lr": 0.008, "patience": 1, "lr_d": 0.45},
}

TARGET_LOAD_COLS = [0, 1, 2]


def season_from_month(month):
    """Return the pre-registered meteorological season for a calendar month."""
    month = int(month)
    if month in (12, 1, 2):
        return 'DJF'
    if month in (3, 4, 5):
        return 'MAM'
    if month in (6, 7, 8):
        return 'JJA'
    if month in (9, 10, 11):
        return 'SON'
    raise ValueError(f'invalid calendar month: {month}')


def _metadata_start_time(metadata_path):
    if not metadata_path or not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    value = metadata.get('time_start')
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def write_forecast_origins(path, data, split_name, metadata_path):
    """Write aligned origin timestamps and fixed season labels for diagnostics."""
    first_target_end = int(data.split_metadata['target_end_index_ranges'][split_name][0])
    n_samples = int(data.split_metadata['sample_counts'][split_name])
    start_time = _metadata_start_time(metadata_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        'sample_index', 'origin_index', 'origin_timestamp', 'origin_season',
        'target_start_index', 'target_start_timestamp', 'target_start_season',
        'target_end_index', 'target_end_timestamp', 'target_end_season',
        'horizon_h', 'split',
    ]
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_index in range(n_samples):
            target_end = first_target_end + sample_index
            target_start = target_end - int(data.h) + 1
            origin = target_start - 1
            def stamp(index):
                return start_time + timedelta(hours=int(index)) if start_time is not None else None
            origin_ts, target_start_ts, target_end_ts = stamp(origin), stamp(target_start), stamp(target_end)
            writer.writerow({
                'sample_index': sample_index,
                'origin_index': origin,
                'origin_timestamp': origin_ts.isoformat() if origin_ts else '',
                'origin_season': season_from_month(origin_ts.month) if origin_ts else '',
                'target_start_index': target_start,
                'target_start_timestamp': target_start_ts.isoformat() if target_start_ts else '',
                'target_start_season': season_from_month(target_start_ts.month) if target_start_ts else '',
                'target_end_index': target_end,
                'target_end_timestamp': target_end_ts.isoformat() if target_end_ts else '',
                'target_end_season': season_from_month(target_end_ts.month) if target_end_ts else '',
                'horizon_h': int(data.h),
                'split': split_name,
            })


def warn_if_legacy_checkpoint_policy(checkpoint_policy):
    if checkpoint_policy == 'legacy':
        warnings.warn(
            "Legacy checkpoint policy is for historical compatibility only: "
            "validation-best and last-epoch checkpoints share model.pt, and the "
            "post-training save overwrites the validation-best weights. Use "
            "--checkpoint_policy best for new experiments.",
            LegacyCheckpointPolicyWarning,
            stacklevel=2,
        )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def output_to_load_shape(output, batch_size, horizon, out_nodes):
    if output.dim() == 4:
        output = output[..., output.shape[-1] // 2]
    return output.reshape(batch_size, horizon, out_nodes)


def output_to_quantile_shape(output, batch_size, horizon, out_nodes):
    if output.dim() != 4:
        raise ValueError(f"expected quantile output [B,H,C,Q], got {output.shape}")
    return output.reshape(batch_size, horizon, out_nodes, output.shape[-1])


def denorm_load_quantiles(data, y, device_flag):
    if device_flag == 'cpu':
        std = data.scale_std[TARGET_LOAD_COLS].reshape(1, 1, len(TARGET_LOAD_COLS), 1)
        mean = data.scale_mean[TARGET_LOAD_COLS].reshape(1, 1, len(TARGET_LOAD_COLS), 1)
        return y * std + mean
    std = torch.from_numpy(data.scale_std[TARGET_LOAD_COLS]).float().to(y.device).view(1, 1, len(TARGET_LOAD_COLS), 1)
    mean = torch.from_numpy(data.scale_mean[TARGET_LOAD_COLS]).float().to(y.device).view(1, 1, len(TARGET_LOAD_COLS), 1)
    return y * std + mean


def pinball_loss(target, prediction, quantiles, median_index=None, label_mask=None):
    target = target.unsqueeze(-1)
    if label_mask is not None:
        valid = label_mask.to(dtype=torch.bool).unsqueeze(-1)
        valid = valid & torch.isfinite(target) & torch.isfinite(prediction).all(dim=-1, keepdim=True)
        # Sanitize masked entries before arithmetic so NaNs cannot poison gradients.
        target = torch.where(valid, target, torch.zeros_like(target))
        prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    else:
        valid = None
    errors = target - prediction
    q = quantiles.to(prediction.device, prediction.dtype).view(1, 1, 1, -1)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    if median_index is not None and prediction.shape[-1] > 1:
        mask = torch.ones(prediction.shape[-1], device=prediction.device, dtype=torch.bool)
        mask[int(median_index)] = False
        loss = loss[..., mask]
    if valid is not None:
        loss = loss.masked_fill(~valid, 0.0)
        denominator = valid.expand_as(loss).sum().clamp_min(1)
        return loss.sum() / denominator
    return loss.mean()


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, horizon, out_nodes, legacy_mape_order=True, label_mask=None):
    model.eval()
    predictions = []
    targets = []
    masks = []

    batches = data.get_batches(X, Y, batch_size, False, masks=label_mask)
    for batch in batches:
        if label_mask is None:
            X, Y = batch
            batch_mask = None
        else:
            X, Y, batch_mask = batch
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        if output.dim() == 4:
            output = output_to_quantile_shape(output, Y.size(0), horizon, out_nodes)[..., model.median_index]
        else:
            output = output_to_load_shape(output, Y.size(0), horizon, out_nodes)
        predictions.append(output)
        targets.append(Y)
        if batch_mask is not None:
            masks.append(batch_mask)

    predict = torch.cat(predictions, dim=0).data.cpu().numpy()
    Ytest = torch.cat(targets, dim=0).data.cpu().numpy()
    predict = data._de_z_score_normalized(predict, 'cpu')
    Ytest = data._de_z_score_normalized(Ytest, 'cpu')
    mask_np = torch.cat(masks, dim=0).cpu().numpy() if masks else None
    if legacy_mape_order:
        # Keep the original Cfc online-eval order for P0 checkpoint reproduction.
        mape = metric_overall(predict, Ytest, mask_np)['mape']
    else:
        # P1+ should select checkpoints by the same reported metric used in saved reports.
        mape = metric_overall(Ytest, predict, mask_np)['mape']
    summary = metric_overall(Ytest, predict, mask_np)
    mae = summary['mae']
    corr = summary['corr']
    return mae, mape, corr


def evaluate_quantile(data, X, Y, model, batch_size, horizon, out_nodes, label_mask=None):
    model.eval()
    predictions = []
    targets = []
    pinball_losses = []
    masks = []

    batches = data.get_batches(X, Y, batch_size, False, masks=label_mask)
    for batch in batches:
        if label_mask is None:
            X, Y = batch
            batch_mask = None
        else:
            X, Y, batch_mask = batch
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
            quantile_output = output_to_quantile_shape(output, Y.size(0), horizon, out_nodes)
            median_output = quantile_output[..., model.median_index]
            loss = pinball_loss(Y, quantile_output, model.quantiles, model.median_index, batch_mask)
        predictions.append(median_output)
        targets.append(Y)
        if batch_mask is not None:
            masks.append(batch_mask)
        pinball_losses.append(loss.item())

    predict = torch.cat(predictions, dim=0).data.cpu().numpy()
    Ytest = torch.cat(targets, dim=0).data.cpu().numpy()
    predict = data._de_z_score_normalized(predict, 'cpu')
    Ytest = data._de_z_score_normalized(Ytest, 'cpu')
    mask_np = torch.cat(masks, dim=0).cpu().numpy() if masks else None
    summary = metric_overall(Ytest, predict, mask_np)
    mae = summary['mae']
    mape = summary['mape']
    corr = summary['corr']
    return mae, mape, corr, float(np.mean(pinball_losses))


def train(data, X, Y_full, model, criterion, optim, batch_size, horizon, out_nodes,
          lambda_conservation=0.0, label_mask=None, max_optimizer_updates=None,
          point_training_recipe=LEGACY_RECIPE, target_tau=None):
    model.train()
    total_loss = 0
    total_data_loss = 0
    total_cons_loss = 0
    total_mae_loss = 0

    iter_count = 0
    batches = data.get_batches(X, Y_full, batch_size, True, masks=label_mask)
    for batch in batches:
        if max_optimizer_updates is not None and optim.step_count >= int(max_optimizer_updates):
            break
        if label_mask is None:
            X, Y_full = batch
            batch_mask = None
        else:
            X, Y_full, batch_mask = batch
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        ty_full = Y_full
        assert ty_full.shape[-1] == out_nodes, f"expected {out_nodes} target columns, got {ty_full.shape}"
        ty = ty_full[:, :, :out_nodes]
        assert ty.shape[-1] == out_nodes, f"expected {out_nodes} load columns, got {ty.shape}"

        raw_output = model(X)
        if raw_output.dim() == 4:
            quantile_output = output_to_quantile_shape(raw_output, ty.size(0), horizon, out_nodes)
            output = quantile_output[..., model.median_index]
        else:
            quantile_output = None
            output = output_to_load_shape(raw_output, ty.size(0), horizon, out_nodes)
            assert output.shape[-1] == out_nodes, f"expected {out_nodes} output columns, got {output.shape}"

        ty = data._de_z_score_normalized(ty, 'gpu')
        output = data._de_z_score_normalized(output, 'gpu')

        if quantile_output is not None:
            loss_data = pinball_loss(ty_full[:, :, :out_nodes], quantile_output, model.quantiles, model.median_index, batch_mask)
        elif point_training_recipe == A3_RECIPE:
            if target_tau is None:
                raise ValueError('A3 point training requires target_tau')
            loss_data = bounded_percentage_loss_physical(ty, output, target_tau, batch_mask)
        else:
            loss_data = mape_loss(ty, output, batch_mask)
        # The new dataset has no physical-constraint loss path.
        loss_cons = output.new_zeros(())
        loss = loss_data
        if batch_mask is None:
            loss_mae = MAE((ty).cpu().detach().numpy(), (output).cpu().detach().numpy())
        else:
            valid = batch_mask.to(dtype=torch.bool)
            loss_mae = torch.abs(ty - output).masked_select(valid).mean().item() if torch.any(valid) else 0.0
        loss.backward()
        total_loss += loss.item()
        total_data_loss += loss_data.item()
        total_cons_loss += loss_cons.item()
        total_mae_loss += float(loss_mae)
        optim.step()

        iter_count += 1
    return (
        total_loss / iter_count,
        total_mae_loss / iter_count,
        total_data_loss / iter_count,
        total_cons_loss / iter_count,
    )


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        _dict = {}
        for _, param in enumerate(model.named_parameters()):
            total_params = param[1].numel()
            k = param[0].split('.')[0]
            if k in _dict.keys():
                _dict[k] += total_params
            else:
                _dict[k] = 0
                _dict[k] += total_params
        total_param = sum(p.numel() for p in model.parameters())
        bytes_per_param = 1
        total_bytes = total_param * bytes_per_param
        total_megabytes = total_bytes / (1024 * 1024)
        return total_param, total_megabytes, _dict


def count_flops(model, input_size):
    def flops_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            H_out, W_out = output.shape[2], output.shape[3]
            K = module.kernel_size[0]
            C_in = module.in_channels
            C_out = module.out_channels
            flops = H_out * W_out * K * K * C_in * C_out
            module.__flops__ += flops

        elif isinstance(module, nn.Linear):
            flops = module.in_features * module.out_features
            module.__flops__ += flops

    for layer in model.modules():
        layer.__flops__ = 0
        layer.register_forward_hook(flops_hook)

    dummy_input = torch.randn(input_size)
    model(dummy_input)

    total_flops = sum(layer.__flops__ for layer in model.modules() if hasattr(layer, '__flops__'))
    return total_flops


def build_parser():
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument(
        '--data',
        type=str,
        default='./data/preprocessed_forecasting_v6/dataset_input.csv',
        help='location of the full local data file; the public repository includes only a schema sample',
    )
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default=None, help='legacy checkpoint path; defaults to model_dir/model.pt')
    parser.add_argument('--model_dir', type=str, default=None, help='directory to save checkpoints')
    parser.add_argument('--result_dir', type=str, default=None, help='directory to save logs and predictions')
    parser.add_argument('--run_tag', type=str, default='', help='optional run tag for logs')
    parser.add_argument('--trial_id', type=str, default='', help='optional HPO trial identifier')
    parser.add_argument('--config_hash', type=str, default='', help='precomputed canonical trial-config SHA256')
    parser.add_argument('--seed', type=int, default=2020, help='random seed for Python, NumPy, and PyTorch')
    parser.add_argument('--model', choices=['cfc', 'itransformer', 'patchtst', 'timemixer', 'timesnet', 'smamba', 'dlinear', 'nlinear', 'cfc_pmd', 'cfc_pmd_cecm', 'cfc_pmd_cecm_quantile'], default='cfc')
    parser.add_argument('--checkpoint_policy', choices=['legacy', 'best', 'last'], default='best')
    parser.add_argument('--split_policy', choices=['legacy', 'embargo'], default='embargo', help='temporal split-window policy')
    parser.add_argument('--evaluation_mode', choices=['validation-only', 'final-test'], default='validation-only', help='whether test tensors may be materialized and evaluated')
    parser.add_argument(
        '--point_training_recipe', '--point-training-recipe',
        choices=[LEGACY_RECIPE, A3_RECIPE], default=LEGACY_RECIPE,
        help='explicit point-training loss/scaler protocol; legacy remains the default',
    )
    parser.add_argument('--metric_protocol', choices=['legacy', 'standard'], default='standard', help='checkpoint-selection MAPE argument order')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--subgraph_size', type=int, default=15, help='k')
    parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
    parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
    parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
    parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')
    parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
    parser.add_argument('--end_channels', type=int, default=64, help='end channels')
    parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
    parser.add_argument('--seq_in_len', type=int, default=24 * 7, help='input sequence length')
    parser.add_argument('--seq_out_len', type=int, default=None, help='output sequence length; defaults to horizon')
    parser.add_argument('--horizon', type=int, choices=[24, 48, 72, 96], default=24)
    parser.add_argument('--use_horizon_preset', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--layers', type=int, default=5, help='number of layers')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
    parser.add_argument('--clip', type=int, default=5, help='clip')
    parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
    parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')
    parser.add_argument('--epochs', type=int, default=None, help='')
    parser.add_argument('--max_optimizer_updates', type=int, default=12000, help='hard upper bound on optimizer.step calls')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--patience', type=int, default=None, help='patience')
    parser.add_argument('--scheduler_patience', type=int, default=None, help='LR scheduler patience; defaults to --patience independently of early stopping')
    parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction, default=False, help='stop after patience validation non-improvements')
    parser.add_argument('--lr_d', type=float, default=None, help='inverse data')
    parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
    parser.add_argument('--step_size', type=int, default=100, help='step_size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--num_nodes', type=int, default=12, help='number of nodes/variables')
    parser.add_argument('--out_nodes', type=int, default=3, help='out_nodes')
    parser.add_argument('--cfc_hidden_size', type=int, default=16, help='CfC recurrent state size')
    parser.add_argument('--cfc_decoder', choices=['flat', 'daily_intraday'], default='flat',
                        help='daily_intraday constrains only the 72h/96h CfC readout')
    parser.add_argument('--period_set', type=int, nargs='+', default=[6, 12, 24], help='non-empty subset of 6, 12, and 24 hour periods')
    parser.add_argument('--d_model', type=int, default=32, help='iTransformer hidden dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='iTransformer attention heads')
    parser.add_argument('--e_layers', type=int, default=1, help='iTransformer encoder layers')
    parser.add_argument('--d_ff', type=int, default=64, help='iTransformer feed-forward dimension')
    parser.add_argument('--factor', type=int, default=1, help='iTransformer attention factor')
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'])
    parser.add_argument('--itransformer_dropout', type=float, default=0.2, help='iTransformer dropout')
    parser.add_argument('--patchtst_patch_len', type=int, default=16, help='PatchTST patch length')
    parser.add_argument('--patchtst_stride', type=int, default=8, help='PatchTST patch stride')
    parser.add_argument('--patchtst_e_layers', type=int, default=2, help='PatchTST encoder layers')
    parser.add_argument('--patchtst_dropout', type=float, default=0.2, help='PatchTST dropout')
    parser.add_argument('--timemixer_num_scales', type=int, default=3, help='TimeMixer multiscale count')
    parser.add_argument('--timemixer_down_factor', type=int, default=2, help='TimeMixer downsample factor')
    parser.add_argument('--timemixer_e_layers', type=int, default=2, help='TimeMixer mixing layers')
    parser.add_argument('--timemixer_dropout', type=float, default=0.2, help='TimeMixer dropout')
    parser.add_argument('--timemixer_decomp_kernel', type=int, default=25, help='TimeMixer decomposition kernel')
    parser.add_argument('--timesnet_e_layers', type=int, default=2, help='TimesNet TimesBlock count')
    parser.add_argument('--timesnet_d_ff', type=int, default=32, help='TimesNet inception channels')
    parser.add_argument('--timesnet_top_k', type=int, default=3, help='TimesNet FFT top-k periods')
    parser.add_argument('--timesnet_num_kernels', type=int, default=3, help='TimesNet inception kernel count')
    parser.add_argument('--timesnet_dropout', type=float, default=0.2, help='TimesNet dropout')
    parser.add_argument('--smamba_e_layers', type=int, default=2, help='S-Mamba SSM layers')
    parser.add_argument('--smamba_d_state', type=int, default=16, help='S-Mamba SSM state size')
    parser.add_argument('--smamba_dropout', type=float, default=0.2, help='S-Mamba dropout')
    parser.add_argument('--smamba_bidirectional', type=int, default=1, help='S-Mamba bidirectional scan (1/0)')
    parser.add_argument('--linear_individual', action='store_true', help='use per-load linear heads for DLinear/NLinear')
    parser.add_argument('--dlinear_kernel_size', type=int, default=25, help='moving-average kernel size for DLinear')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=8, help='epochs to freeze Cfc backbone for cfc_pmd')
    parser.add_argument('--best_epoch_min_offset', type=int, default=0, help='earliest epoch eligible to supply best.pt; 0 keeps the historical behaviour in which an untrained stage can win')
    parser.add_argument('--pmd_lr', type=float, default=1e-3, help='PMD head AdamW learning rate')
    parser.add_argument('--pmd_backbone_lr', type=float, default=1e-4, help='Cfc backbone AdamW learning rate for cfc_pmd')
    parser.add_argument('--pmd_weight_decay', type=float, default=0.0, help='PMD head weight decay')
    parser.add_argument('--pmd_backbone_weight_decay', type=float, default=0.0, help='Cfc backbone weight decay for cfc_pmd')
    parser.add_argument('--pmd_gate_init', type=float, default=-3.0, help='initial PMD residual gate logit')
    parser.add_argument('--pmd_head_type', choices=['linear', 'mlp'], default='linear', help='PMD residual head type')
    parser.add_argument('--pmd_hidden_dim', type=int, default=64, help='PMD MLP hidden dimension')
    parser.add_argument('--pmd_dropout', type=float, default=0.1, help='PMD MLP dropout')
    parser.add_argument('--pmd_fine_mode', choices=['direct', 'lowrank'], default='direct', help='PMD fine residual parameterization')
    parser.add_argument('--pmd_fine_rank', type=int, default=2, help='rank for lowrank PMD fine residual')
    parser.add_argument('--cfc_backbone_path', type=str, default=None, help='optional Cfc checkpoint to initialize cfc_pmd backbone')
    parser.add_argument('--pmd_checkpoint_path', type=str, default=None, help='optional PMD checkpoint to initialize cfc_pmd_cecm')
    parser.add_argument('--checkpoint_init_policy', choices=['auto', 'random', 'required'], default='auto', help='checkpoint initialization policy for composed CfC models')
    parser.add_argument('--lr_mult_cfc', type=float, default=1.0, help='multiplier for an existing CfC optimizer group')
    parser.add_argument('--lr_mult_pmd', type=float, default=1.0, help='multiplier for an existing PMD optimizer group')
    parser.add_argument('--lr_mult_cecm', type=float, default=1.0, help='multiplier for an existing CECM optimizer group')
    parser.add_argument('--cecm_gate_init', type=float, default=-3.0, help='initial CECM residual gate logit')
    parser.add_argument('--cecm_conditioning', choices=['static', 'linear_lead', 'linear_lead_season'], default='static',
                        help='static correction or opt-in forecast-lead/origin-calendar conditioning')
    parser.add_argument('--cecm_lr', type=float, default=1e-3, help='CECM AdamW learning rate')
    parser.add_argument('--cecm_pmd_lr', type=float, default=1e-4, help='PMD fine-tune learning rate for cfc_pmd_cecm')
    parser.add_argument('--cecm_cfc_lr', type=float, default=5e-5, help='optional Cfc fine-tune learning rate for cfc_pmd_cecm')
    parser.add_argument('--cecm_weight_decay', type=float, default=0.0, help='CECM weight decay')
    parser.add_argument('--cecm_pmd_train_start_epoch', type=int, default=6, help='epoch to start training PMD with CECM')
    parser.add_argument('--cecm_cfc_train_start_epoch', type=int, default=0, help='epoch to start Cfc fine-tuning for CECM; 0 keeps Cfc frozen')
    parser.add_argument('--cecm_offdiag_only', action='store_true', help='learn only off-diagonal CECM residual mixing')
    parser.add_argument('--freeze_cecm', action='store_true', help='freeze CECM parameters for a PMD-only continuation ablation')
    parser.add_argument('--cecm_checkpoint_path', type=str, default=None, help='optional full CECM checkpoint to initialize cfc_pmd_cecm')
    parser.add_argument('--lambda_conservation', type=float, default=0.0, help='deprecated compatibility flag; must remain 0 for the new dataset')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.1, 0.5, 0.9], help='quantile levels for calibration wrapper')
    parser.add_argument('--quantile_lr', type=float, default=1e-3, help='quantile spread-head AdamW learning rate')
    parser.add_argument('--quantile_weight_decay', type=float, default=0.0, help='quantile spread-head weight decay')
    parser.add_argument('--quantile_hidden_dim', type=int, default=32, help='quantile spread-head hidden dimension')
    parser.add_argument('--quantile_dropout', type=float, default=0.1, help='quantile spread-head dropout')
    parser.add_argument('--quantile_fine_rank', type=int, default=2, help='quantile intraday spread rank')
    parser.add_argument('--quantile_spread_init', type=float, default=0.2, help='initial normalized spread around q50')
    parser.add_argument('--point_checkpoint_path', type=str, default=None, help='full point checkpoint for cfc_pmd_cecm_quantile')
    parser.add_argument('--selected_point_model', type=str, default=None, help='frozen point-path registry id used by the P4 wrapper')
    return parser


def finalize_args(args):
    preset = HORIZON_PRESETS[args.horizon]
    if args.seq_out_len is None:
        args.seq_out_len = args.horizon
    if args.use_horizon_preset:
        for key, value in preset.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        args.horizon_preset_applied = True
        args.training_control_source = 'horizon_preset_or_explicit_override'
    else:
        required = ('epochs', 'lr', 'patience', 'lr_d')
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(
                '--no-use_horizon_preset requires explicit horizon-independent training controls: '
                + ', '.join(f'--{name}' for name in missing)
            )
        args.horizon_preset_applied = False
        args.training_control_source = 'explicit_cli_no_horizon_preset'
    if args.scheduler_patience is None:
        args.scheduler_patience = args.patience
    if args.scheduler_patience < 0:
        raise ValueError('--scheduler_patience must be non-negative')
    if not args.period_set or any(period not in {6, 12, 24} for period in args.period_set):
        raise ValueError('--period_set must be a non-empty subset of 6, 12, and 24')
    if len(set(args.period_set)) != len(args.period_set):
        raise ValueError('--period_set must not contain duplicates')
    args.period_set = sorted(args.period_set)
    for name in ('lr_mult_cfc', 'lr_mult_pmd', 'lr_mult_cecm'):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name} must be positive')
    if args.evaluation_mode == 'validation-only' and args.checkpoint_policy != 'best':
        raise ValueError('--evaluation_mode validation-only requires --checkpoint_policy best')
    if args.lambda_conservation != 0.0:
        raise ValueError('--lambda_conservation must be 0: physical constraint is disabled for the new dataset')
    if args.model == 'cfc_pmd_cecm_quantile':
        if args.selected_point_model is None:
            args.selected_point_model = 'p2b'
        if args.selected_point_model not in {'p0', 'p2a', 'p2b'}:
            raise ValueError('--selected_point_model must be one of p0, p2a, or p2b for P4')
        if args.checkpoint_init_policy == 'required' and not args.point_checkpoint_path:
            raise ValueError('P4 with checkpoint_init_policy=required needs --point_checkpoint_path')
    if args.max_optimizer_updates <= 0:
        raise ValueError('--max_optimizer_updates must be positive')
    args.actual_lr_single = None
    args.actual_lr_cfc = None
    args.actual_lr_pmd = None
    args.actual_lr_cecm = None
    args.actual_lr_quantile = None
    if args.model == 'cfc_pmd':
        args.actual_lr_cfc = args.pmd_backbone_lr * args.lr_mult_cfc
        args.actual_lr_pmd = args.pmd_lr * args.lr_mult_pmd
    elif args.model == 'cfc_pmd_cecm':
        args.actual_lr_cfc = args.cecm_cfc_lr * args.lr_mult_cfc
        args.actual_lr_pmd = args.cecm_pmd_lr * args.lr_mult_pmd
        args.actual_lr_cecm = args.cecm_lr * args.lr_mult_cecm
    elif args.model == 'cfc_pmd_cecm_quantile':
        args.actual_lr_quantile = args.quantile_lr
    else:
        args.actual_lr_single = args.lr
    if args.result_dir is None:
        run_roots = {
            'cfc': 'p0-unfixed-cfc',
            'itransformer': 'sota-itransformer',
            'patchtst': 'sota-patchtst',
            'timemixer': 'sota-timemixer',
            'timesnet': 'sota-timesnet',
            'smamba': 'sota-smamba',
            'dlinear': 'p1c-dlinear',
            'nlinear': 'p1c-nlinear',
            'cfc_pmd': 'p2a-cfc-pmd',
            'cfc_pmd_cecm': 'p2b-cfc-pmd-cecm',
            'cfc_pmd_cecm_quantile': 'p4-cfc-pmd-cecm-quantile',
        }
        run_root = run_roots[args.model]
        args.result_dir = f'./result/{run_root}/{args.horizon}-steps'
    if args.model_dir is None:
        run_roots = {
            'cfc': 'p0-unfixed-cfc',
            'itransformer': 'sota-itransformer',
            'patchtst': 'sota-patchtst',
            'timemixer': 'sota-timemixer',
            'timesnet': 'sota-timesnet',
            'smamba': 'sota-smamba',
            'dlinear': 'p1c-dlinear',
            'nlinear': 'p1c-nlinear',
            'cfc_pmd': 'p2a-cfc-pmd',
            'cfc_pmd_cecm': 'p2b-cfc-pmd-cecm',
            'cfc_pmd_cecm_quantile': 'p4-cfc-pmd-cecm-quantile',
        }
        run_root = run_roots[args.model]
        args.model_dir = f'./model/{run_root}/{args.horizon}-steps'
    ensure_dir(args.result_dir)
    ensure_dir(args.model_dir)
    if args.save is None:
        args.save = os.path.join(args.model_dir, 'model.pt')
    return args


def build_hparams(args):
    return {
        "optimizer": "adam",
        "base_lr": 0.05,
        "decay_lr": 0.95,
        "backbone_activation": "lecun",
        "forget_bias": 2.4,
        "epochs": 80,
        "class_weight": 8,
        "clipnorm": 0,
        "hidden_size": args.cfc_hidden_size,
        "backbone_units": args.cfc_hidden_size,
        "backbone_dr": args.dropout,
        "backbone_layers": 2,
        "weight_decay": 0,
        "optim": "adamw",
        "init": 0.53,
        "batch_size": 64,
        "out_lens": args.seq_out_len,
        "cfc_decoder": getattr(args, "cfc_decoder", "flat"),
        "in_lens": 168,
        "period_len": list(args.period_set),
        "sd_kernel_size": [6, 24],
        "use_mixed": False,
        "no_gate": False,
        "minimal": False,
        "use_ltc": False,
    }


def build_cfc(args, hparams):
    return Cfc(
            in_features=args.num_nodes,
            hidden_size=hparams["hidden_size"],
            out_feature=args.out_nodes,
            return_sequences=True,
            hparams=hparams,
            use_mixed=hparams["use_mixed"],
        use_ltc=hparams["use_ltc"],
    )


def build_quantile_point_model(args, hparams):
    """Build the frozen point-path architecture selected for P4.

    P4 is a wrapper, not a synonym for P2b: the frozen point path may be P0,
    P2a, or P2b.  Keeping this construction explicit makes the checkpoint
    shape and the recorded ``selected_point_model`` agree.
    """
    selected = args.selected_point_model or 'p2b'
    if selected == 'p0':
        return build_cfc(args, hparams)
    pmd_model = CfcWithPMD(
        build_cfc(args, hparams),
        pred_len=args.seq_out_len,
        c_out=args.out_nodes,
        gate_init=args.pmd_gate_init,
        head_type=args.pmd_head_type,
        hidden_dim=args.pmd_hidden_dim,
        dropout=args.pmd_dropout,
        fine_mode=args.pmd_fine_mode,
        fine_rank=args.pmd_fine_rank,
    )
    if selected == 'p2a':
        return pmd_model
    if selected == 'p2b':
        return CfcWithPMDCECM(
            pmd_model,
            c_out=args.out_nodes,
            gate_init=args.cecm_gate_init,
            offdiag_only=args.cecm_offdiag_only,
            conditioning=getattr(args, 'cecm_conditioning', 'static'),
        )
    raise ValueError(
        "--selected_point_model must be one of p0, p2a, or p2b for P4"
    )


def candidate_cfc_backbone_paths(args):
    if args.cfc_backbone_path:
        return [args.cfc_backbone_path]
    return [
        os.path.join('model', 'p0-unfixed-cfc', f'{args.horizon}-steps', 'best.pt'),
        os.path.join('model', 'p0-unfixed-cfc', f'{args.horizon}-steps', 'model.pt'),
        os.path.join('model', 'p0-fixed-cfc', f'{args.horizon}-steps', 'best.pt'),
    ]


def candidate_pmd_backbone_paths(args):
    if args.pmd_checkpoint_path:
        return [args.pmd_checkpoint_path]
    return [
        os.path.join('model', 'p2a-cfc-pmd-compressed', f'{args.horizon}-steps', 'best.pt'),
        os.path.join('model', 'p2a-cfc-pmd', f'{args.horizon}-steps', 'best.pt'),
    ]


def candidate_cecm_checkpoint_paths(args):
    if args.cecm_checkpoint_path:
        return [args.cecm_checkpoint_path]
    if args.lambda_conservation > 0 or args.cecm_checkpoint_path:
        return [
            os.path.join('model', 'p2b-cfc-pmd-cecm-only', f'{args.horizon}-steps', 'best.pt'),
        ]
    return []


def candidate_point_checkpoint_paths(args):
    if args.point_checkpoint_path:
        return [args.point_checkpoint_path]
    return [
        os.path.join('model', 'p2b-cfc-pmd-cecm-only', f'{args.horizon}-steps', 'best.pt'),
    ]


def load_checkpoint_cpu(path):
    with open(path, 'rb') as f:
        try:
            return torch.load(f, weights_only=False, map_location='cpu')
        except TypeError:
            f.seek(0)
            return torch.load(f, map_location='cpu')


def try_load_cfc_backbone(model, args):
    if args.checkpoint_init_policy == 'random':
        if args.model == 'cfc_pmd':
            model.backbone_source = 'random_init'
        elif args.model == 'cfc_pmd_cecm':
            model.pmd_source = 'random_init'
            model.cecm_source = 'random_init'
        elif args.model == 'cfc_pmd_cecm_quantile':
            model.point_source = 'random_init'
        return

    if args.model == 'cfc_pmd':
        for path in candidate_cfc_backbone_paths(args):
            if not os.path.exists(path):
                continue
            checkpoint = load_checkpoint_cpu(path)
            if hasattr(checkpoint, 'state_dict'):
                model.backbone.load_state_dict(checkpoint.state_dict())
                model.backbone_source = path
                return
            if isinstance(checkpoint, dict):
                model.backbone.load_state_dict(checkpoint)
                model.backbone_source = path
                return
        if args.checkpoint_init_policy == 'required':
            raise FileNotFoundError('no compatible CfC backbone checkpoint found')
        model.backbone_source = 'random_init'
        return

    if args.model in ['cfc_pmd_cecm', 'cfc_pmd_cecm_quantile']:
        target_model = model if args.model == 'cfc_pmd_cecm' else model.point_model
        if args.model == 'cfc_pmd_cecm_quantile':
            for path in candidate_point_checkpoint_paths(args):
                if not os.path.exists(path):
                    continue
                checkpoint = load_checkpoint_cpu(path)
                state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                if isinstance(state_dict, dict):
                    try:
                        target_model.load_state_dict(state_dict)
                    except RuntimeError:
                        continue
                    model.point_source = path
                    return
            if args.checkpoint_init_policy == 'required':
                raise FileNotFoundError('no compatible point checkpoint found')
            model.point_source = 'random_init'
            return

        for path in candidate_cecm_checkpoint_paths(args):
            if not os.path.exists(path):
                continue
            checkpoint = load_checkpoint_cpu(path)
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            if isinstance(state_dict, dict):
                try:
                    target_model.load_state_dict(state_dict)
                except RuntimeError:
                    continue
                model.pmd_source = getattr(checkpoint, 'pmd_source', path)
                model.pmd.backbone_source = getattr(checkpoint, 'backbone_source', model.pmd.backbone_source)
                model.cecm_source = path
                return
        for path in candidate_pmd_backbone_paths(args):
            if not os.path.exists(path):
                continue
            checkpoint = load_checkpoint_cpu(path)
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
            if isinstance(state_dict, dict):
                try:
                    model.pmd.load_state_dict(state_dict)
                except RuntimeError:
                    continue
                model.pmd_source = path
                model.pmd.backbone_source = getattr(checkpoint, 'backbone_source', model.pmd.backbone_source)
                model.cecm_source = 'random_init'
                return
        if args.checkpoint_init_policy == 'required':
            raise FileNotFoundError('no compatible PMD or CECM checkpoint found')
        model.pmd_source = 'random_init'
        model.cecm_source = 'random_init'


def configure_cecm_calendar(model, args, data):
    point = getattr(model, 'point_model', model)
    if getattr(point, 'cecm_conditioning', 'static') != 'linear_lead_season':
        return None
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(args.data)), 'preprocessing_metadata.json')
    with open(metadata_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    with open(args.data, encoding='utf-8-sig', newline='') as handle:
        columns = next(csv.reader(handle), [])
    if (columns != metadata.get('model_columns') or len(columns) != data.m
            or len(set(columns)) != len(columns)):
        raise ValueError('calendar feature schema differs between input and preprocessing metadata')
    names = ['DayOfYear_sin', 'DayOfYear_cos']
    if any(name not in columns for name in names):
        raise ValueError('seasonal CECM requires observed origin calendar columns')
    indices = [columns.index(name) for name in names]
    point.configure_season_calendar(indices, data.scale_mean[indices], data.scale_std[indices])
    return {
        'feature_names': names,
        'column_indices': indices,
        'normalization_mean': point.cecm_calendar_mean.detach().cpu().tolist(),
        'normalization_scale': point.cecm_calendar_scale.detach().cpu().tolist(),
        'fit_end_exclusive': int(data.train_size * data.n),
        'source': 'last_observed_input_hour',
        'phase_definition': 'sin_cos(2*pi*(day_of_year-1)/365.25)',
        'held_constant_across_forecast': True,
        'future_observations_used': False,
    }


def build_model(args, hparams):
    if args.model == 'cfc':
        return build_cfc(args, hparams)
    if args.model == 'itransformer':
        return ITransformerBaseline(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            dropout=args.itransformer_dropout,
            factor=args.factor,
            activation=args.activation,
        )
    if args.model == 'patchtst':
        return PatchTSTBaseline(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.patchtst_e_layers,
            d_ff=args.d_ff,
            dropout=args.patchtst_dropout,
            patch_len=args.patchtst_patch_len,
            stride=args.patchtst_stride,
            activation=args.activation,
        )
    if args.model == 'timemixer':
        return TimeMixerBaseline(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            d_model=args.d_model,
            e_layers=args.timemixer_e_layers,
            dropout=args.timemixer_dropout,
            num_scales=args.timemixer_num_scales,
            down_factor=args.timemixer_down_factor,
            decomp_kernel=args.timemixer_decomp_kernel,
        )
    if args.model == 'timesnet':
        return TimesNetBaseline(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            d_model=args.d_model,
            e_layers=args.timesnet_e_layers,
            d_ff=args.timesnet_d_ff,
            dropout=args.timesnet_dropout,
            top_k=args.timesnet_top_k,
            num_kernels=args.timesnet_num_kernels,
        )
    if args.model == 'smamba':
        return SMambaBaseline(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            d_model=args.d_model,
            e_layers=args.smamba_e_layers,
            d_state=args.smamba_d_state,
            dropout=args.smamba_dropout,
            bidirectional=bool(args.smamba_bidirectional),
        )
    if args.model == 'dlinear':
        return DLinear(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            individual=args.linear_individual,
            kernel_size=args.dlinear_kernel_size,
        )
    if args.model == 'nlinear':
        return NLinear(
            seq_len=args.seq_in_len,
            pred_len=args.seq_out_len,
            enc_in=args.num_nodes,
            c_out=args.out_nodes,
            individual=args.linear_individual,
        )
    if args.model == 'cfc_pmd':
        return CfcWithPMD(
            build_cfc(args, hparams),
            pred_len=args.seq_out_len,
            c_out=args.out_nodes,
            gate_init=args.pmd_gate_init,
            head_type=args.pmd_head_type,
            hidden_dim=args.pmd_hidden_dim,
            dropout=args.pmd_dropout,
            fine_mode=args.pmd_fine_mode,
            fine_rank=args.pmd_fine_rank,
        )
    if args.model == 'cfc_pmd_cecm':
        pmd_model = CfcWithPMD(
            build_cfc(args, hparams),
            pred_len=args.seq_out_len,
            c_out=args.out_nodes,
            gate_init=args.pmd_gate_init,
            head_type=args.pmd_head_type,
            hidden_dim=args.pmd_hidden_dim,
            dropout=args.pmd_dropout,
            fine_mode=args.pmd_fine_mode,
            fine_rank=args.pmd_fine_rank,
        )
        return CfcWithPMDCECM(
            pmd_model,
            c_out=args.out_nodes,
            gate_init=args.cecm_gate_init,
            offdiag_only=args.cecm_offdiag_only,
            cecm_trainable=not args.freeze_cecm,
            conditioning=getattr(args, 'cecm_conditioning', 'static'),
        )
    if args.model == 'cfc_pmd_cecm_quantile':
        point_model = build_quantile_point_model(args, hparams)
        return CfcPMDCECMQuantile(
            point_model,
            pred_len=args.seq_out_len,
            c_out=args.out_nodes,
            quantiles=args.quantiles,
            hidden_dim=args.quantile_hidden_dim,
            dropout=args.quantile_dropout,
            fine_rank=args.quantile_fine_rank,
            spread_init=args.quantile_spread_init,
        )
    raise ValueError(f"unknown model {args.model}")


def build_optimizer(args, model, steps_per_epoch):
    scheduler_patience = getattr(args, 'scheduler_patience', None)
    if scheduler_patience is None:
        scheduler_patience = args.patience
    if args.model == 'cfc_pmd':
        return MultiGroupPlateauOptim(
            [
                {
                    "params": model.backbone_parameters(),
                    "lr": args.actual_lr_cfc,
                    "weight_decay": args.pmd_backbone_weight_decay,
                },
                {
                    "params": model.pmd_parameters(),
                    "lr": args.actual_lr_pmd,
                    "weight_decay": args.pmd_weight_decay,
                },
            ],
            args.clip,
            'min',
            args.lr_d,
            scheduler_patience,
        )

    if args.model == 'cfc_pmd_cecm':
        return MultiGroupPlateauOptim(
            [
                {
                    "params": model.cfc_parameters(),
                    "lr": args.actual_lr_cfc,
                    "weight_decay": args.pmd_backbone_weight_decay,
                },
                {
                    "params": model.pmd_parameters(),
                    "lr": args.actual_lr_pmd,
                    "weight_decay": args.pmd_weight_decay,
                },
                {
                    "params": model.cecm_parameters(),
                    "lr": args.actual_lr_cecm,
                    "weight_decay": args.cecm_weight_decay,
                },
            ],
            args.clip,
            'min',
            args.lr_d,
            scheduler_patience,
        )

    if args.model == 'cfc_pmd_cecm_quantile':
        model.set_point_trainable(False)
        return MultiGroupPlateauOptim(
            [
                {
                    "params": model.quantile_parameters(),
                    "lr": args.actual_lr_quantile,
                    "weight_decay": args.quantile_weight_decay,
                },
            ],
            args.clip,
            'min',
            args.lr_d,
            scheduler_patience,
        )

    return Optim(
        model.parameters(), args.optim, args.actual_lr_single, args.clip, 'min', args.lr_d, scheduler_patience, 1, args.epochs,
        lr_decay=args.weight_decay
    )


def fix_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def log_line(args, message):
    with open(os.path.join(args.result_dir, 'data.txt'), 'a', encoding='utf-8') as f:
        print(message, flush=True, file=f)


def write_run_config(args, hparams=None):
    config_path = os.path.join(args.result_dir, 'config.json')
    payload = dict(vars(args))
    payload.update({
        'input_dim': 12,
        'target_columns': ['KW', 'CHWTON', 'HTmmBTU'],
        'physics_constraint': 'disabled',
        'combined_mmbtu_input': False,
        'point_training_recipe': args.point_training_recipe,
        'target_loss': (
            'target_wise_bounded_percentage_physical_space'
            if args.point_training_recipe == A3_RECIPE else 'legacy_mape_epsilon_1e-2'
        ),
        'target_scaler': (
            'training_only_median_iqr_over_1.349'
            if args.point_training_recipe == A3_RECIPE else 'training_only_mean_std'
        ),
        'retain_zero_labels': True,
        'retain_extreme_finite_labels': True,
        'clip_or_winsorize_targets': False,
        'test_materialized': bool(getattr(args, 'evaluation_mode', '') == 'final-test'),
        'test_accessed': bool(getattr(args, 'evaluation_mode', '') == 'final-test'),
    })
    data_path = os.path.abspath(args.data)
    if os.path.isfile(data_path):
        digest = hashlib.sha256()
        with open(data_path, 'rb') as data_handle:
            for chunk in iter(lambda: data_handle.read(1024 * 1024), b''):
                digest.update(chunk)
        payload['data_sha256'] = digest.hexdigest()
    if hparams is not None:
        payload['effective_hparams'] = hparams
    with open(config_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def write_validation_history(args, history):
    with open(os.path.join(args.result_dir, 'validation_history.json'), 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
        handle.write('\n')


def write_split_manifest(args, data, device):
    cuda_active = device.type == 'cuda' and torch.cuda.is_available()
    payload = {
        'evaluation_mode': args.evaluation_mode,
        'metric_protocol': args.metric_protocol,
        'checkpoint_policy': args.checkpoint_policy,
        'python_executable': sys.executable,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_name': torch.cuda.get_device_name(device) if cuda_active else None,
        'data_path': os.path.abspath(args.data),
        'input_dim': int(data.m),
        'out_nodes': int(args.out_nodes),
        'target_columns': ['KW', 'CHWTON', 'HTmmBTU'],
        'physics_constraint': 'disabled',
        'combined_mmbtu_input': False,
        'point_training_recipe': args.point_training_recipe,
        'target_scaler_sha256': getattr(args, 'target_scaler_sha256', None),
        'target_tau_sha256': getattr(args, 'target_tau_sha256', None),
        'a3_protocol_sha256': getattr(args, 'a3_protocol_sha256', None),
        'test_materialized': data.test is not None,
        'test_accessed': args.evaluation_mode == 'final-test',
        **data.split_metadata,
    }
    with open(os.path.join(args.result_dir, 'split_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def build_metrics_payload(y_true, prediction, horizon, split_name, label_mask=None):
    return {
        'split': split_name,
        'shape': list(y_true.shape),
        'horizon': int(horizon),
        'metric_semantics': {
            'mape': 'mean(abs((y_true - y_pred) / y_true)) * 100; no epsilon floor',
            'corr': 'mean Pearson-style correlation over non-constant axes',
        },
        'label_valid_count': int(np.asarray(label_mask, dtype=bool).sum()) if label_mask is not None else int(np.asarray(y_true).size),
        'label_total_count': int(np.asarray(y_true).size),
        'overall': metric_overall(y_true, prediction, label_mask),
        'by_load': metric_by_load(y_true, prediction, mask=label_mask),
        'by_horizon': metric_by_horizon(y_true, prediction, label_mask),
        'by_load_and_horizon': metric_by_load_and_horizon(y_true, prediction, mask=label_mask),
    }


def save_prediction_bundle(args, model, prefix, split_name, y_true, prediction, quantiles, label_mask=None):
    y_true_np = y_true.cpu().numpy()
    prediction_np = prediction.cpu().numpy()
    np.save(os.path.join(args.result_dir, f'{prefix}_y_true.npy'), y_true_np)
    np.save(os.path.join(args.result_dir, f'{prefix}_predict_value.npy'), prediction_np)
    if label_mask is not None:
        np.save(os.path.join(args.result_dir, f'{prefix}_label_valid_mask.npy'), label_mask.cpu().numpy().astype(np.uint8))

    if prefix == 'all':
        torch.save(y_true, os.path.join(args.result_dir, 'all_y_true.pt'))
        torch.save(prediction, os.path.join(args.result_dir, 'all_predict_value.pt'))

    mask_np = None if label_mask is None else label_mask.cpu().numpy().astype(bool)
    metrics = build_metrics_payload(y_true_np, prediction_np, args.horizon, split_name, mask_np)
    metrics['test_materialized'] = args.evaluation_mode == 'final-test'
    metrics['test_accessed'] = args.evaluation_mode == 'final-test'
    metrics_name = 'metrics_full.json' if prefix == 'all' else f'{prefix}_metrics_full.json'
    with open(os.path.join(args.result_dir, metrics_name), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
        handle.write('\n')

    if quantiles is not None:
        quantiles_np = quantiles.cpu().numpy()
        np.save(os.path.join(args.result_dir, f'{prefix}_quantile_value.npy'), quantiles_np)
        if prefix == 'all':
            torch.save(quantiles, os.path.join(args.result_dir, 'all_quantile_value.pt'))
        metadata_name = 'quantile_metadata.json' if prefix == 'all' else f'{prefix}_quantile_metadata.json'
        model.write_quantile_metadata(os.path.join(args.result_dir, metadata_name))
    return metrics


def get_cecm_module(model):
    """Unwrap the frozen point model when a quantile wrapper is supplied."""
    candidate = getattr(model, 'point_model', model)
    return candidate if hasattr(candidate, 'cecm_parameter_snapshot') else None


def write_cecm_parameter_snapshot(model, path, args=None):
    """Write the evaluated CECM parameters with their conditioning schema."""
    cecm = get_cecm_module(model)
    if cecm is None:
        return None
    payload = cecm.cecm_parameter_snapshot()
    payload.update({
        'model': getattr(args, 'model', None) if args is not None else None,
        'horizon': int(getattr(args, 'horizon', cecm.pmd.pred_len)) if args is not None else int(cecm.pmd.pred_len),
        'seed': int(getattr(args, 'seed', -1)) if args is not None else None,
        'checkpoint_policy': getattr(args, 'checkpoint_policy', None) if args is not None else None,
        'source': 'evaluated_checkpoint',
    })
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return payload


def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        torch.save(model, f)


def load_model(path, map_location=None):
    with open(path, 'rb') as f:
        try:
            return torch.load(f, weights_only=False, map_location=map_location)
        except TypeError:
            f.seek(0)
            return torch.load(f, map_location=map_location)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def formal_code_hashes():
    root = os.path.dirname(os.path.abspath(__file__))
    relatives = [
        'train.py', 'training_protocols.py', 'util.py', 'trainer.py',
        'torch_cfc.py', 'cfc_pci_heads.py', 'metrics.py',
        os.path.join('scripts', 'run_new_dataset_hpo.py'),
        os.path.join('scripts', 'run_new_dataset_validation.py'),
        os.path.join('scripts', 'new_dataset_protocol.py'),
    ]
    return {
        relative.replace('\\', '/'): _sha256_file(os.path.join(root, relative))
        for relative in relatives
        if os.path.isfile(os.path.join(root, relative))
    }


def write_a3_artifacts(args, scaler, target_tau, protocol):
    if args.point_training_recipe != A3_RECIPE:
        return
    payloads = {
        'scaler.json': scaler,
        'target_tau.json': target_tau,
        'a3_protocol.json': protocol,
        'environment.json': {
            'python_executable': sys.executable,
            'python_version': sys.version,
            'platform': platform.platform(),
            'torch_version': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'device': args.device,
            'cuda_device_name': torch.cuda.get_device_name(torch.device(args.device)),
        },
        'code_hashes.json': formal_code_hashes(),
    }
    for name, payload in payloads.items():
        with open(os.path.join(args.result_dir, name), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
    config_path = os.path.join(args.result_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as handle:
        resolved = json.load(handle)
    resolved.update({
        'point_training_recipe': A3_RECIPE,
        'target_scaler_sha256': scaler['scaler_sha256'],
        'target_tau_sha256': target_tau['target_tau_sha256'],
        'a3_protocol_sha256': protocol['a3_protocol_sha256'],
    })
    with open(config_path, 'w', encoding='utf-8') as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
        handle.write('\n')
    with open(os.path.join(args.result_dir, 'resolved_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
        handle.write('\n')


def write_epoch_metrics_csv(args, history):
    path = os.path.join(args.result_dir, 'epoch_metrics.csv')
    fields = [
        'epoch', 'epoch_seconds', 'train_mape_loss', 'train_mae_loss',
        'validation_mae', 'validation_mape', 'validation_corr',
        'validation_pinball', 'selection_metric', 'improved', 'stale_epochs',
        'trainable_stage', 'stage_start_epoch', 'stage_stale_epochs',
        'optimizer_group_lrs_used', 'optimizer_group_lrs',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({
                **row,
                'optimizer_group_lrs_used': json.dumps(row.get('optimizer_group_lrs_used', [])),
                'optimizer_group_lrs': json.dumps(row.get('optimizer_group_lrs', [])),
            })


def write_prediction_distribution(args, prediction, label_mask):
    values = prediction.detach().cpu().numpy()
    mask = label_mask.detach().cpu().numpy().astype(bool) if label_mask is not None else np.isfinite(values)
    names = ('KW', 'CHWTON', 'HTmmBTU')
    loads = {}
    for index, name in enumerate(names):
        valid = mask[..., index] & np.isfinite(values[..., index])
        selected = values[..., index][valid]
        negative = selected < 0
        loads[name] = {
            'min': float(np.min(selected)), 'max': float(np.max(selected)),
            'p01': float(np.percentile(selected, 1)), 'p99': float(np.percentile(selected, 99)),
            'mean': float(np.mean(selected)), 'std': float(np.std(selected)),
            'negative_count': int(negative.sum()),
            'negative_rate': float(negative.mean()) if selected.size else 0.0,
            'valid_label_count': int(valid.sum()),
            'invalid_label_count': int(valid.size - valid.sum()),
        }
    valid_all = mask & np.isfinite(values)
    payload = {
        'unit': 'kW-equivalent',
        'loads': loads,
        'negative_count': int(((values < 0) & valid_all).sum()),
        'negative_rate': float(((values < 0) & valid_all).sum() / valid_all.sum()) if valid_all.any() else 0.0,
    }
    with open(os.path.join(args.result_dir, 'prediction_distribution.json'), 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def main():
    run_start_time = time.time()
    training_start_time = time.perf_counter()
    args = finalize_args(build_parser().parse_args())
    warn_if_legacy_checkpoint_policy(args.checkpoint_policy)
    print("============Preparation==================")
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False')
    fix_seed(args.seed, use_cuda=device.type == 'cuda')
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    write_run_config(args)
    torch.set_num_threads(3)

    train_mae_losses = []
    val_mae_losses = []
    Data = DataLoaderS(
        args.data,
        0.8,
        0.1,
        device,
        args.horizon,
        args.seq_in_len,
        args.normalize,
        split_policy=args.split_policy,
        materialize_test=args.evaluation_mode == 'final-test',
    )

    a3_scaler = None
    a3_target_tau = None
    a3_protocol = None
    if args.point_training_recipe == A3_RECIPE:
        train_end = int(0.8 * Data.n)
        a3_scaler = fit_target_robust_scaler(Data.raw_labels, Data.label_valid_mask, train_end)
        a3_target_tau = fit_target_bounded_tau(Data.raw_labels, Data.label_valid_mask, train_end)
        apply_target_robust_scaler(Data, a3_scaler)
        a3_protocol = build_a3_protocol_manifest(
            a3_scaler,
            a3_target_tau,
            test_materialized=Data.test is not None,
            test_accessed=args.evaluation_mode == 'final-test',
        )
        args.target_scaler_sha256 = a3_scaler['scaler_sha256']
        args.target_tau_sha256 = a3_target_tau['target_tau_sha256']
        args.a3_protocol_sha256 = a3_protocol['a3_protocol_sha256']

    # Validate input dimensions match model expectations
    actual_input_dim = Data.m
    if actual_input_dim != args.num_nodes:
        raise ValueError(
            f"Data input dimension mismatch: dataset has {actual_input_dim} columns "
            f"but model expects {args.num_nodes}. Check preprocessing output and --num_nodes."
        )
    target_dim = int(Data.label_dat.shape[1])
    if target_dim != args.out_nodes:
        raise ValueError(
            f"Target dimension mismatch: dataset exposes {target_dim} target columns but model expects {args.out_nodes} (KW, CHWTON, HTmmBTU)."
        )

    write_split_manifest(args, Data, device)
    hparams = build_hparams(args)
    write_run_config(args, hparams=hparams)
    write_a3_artifacts(args, a3_scaler, a3_target_tau, a3_protocol)
    model = build_model(args, hparams)
    try_load_cfc_backbone(model, args)
    calendar_binding = configure_cecm_calendar(model, args, Data)
    if calendar_binding is not None:
        args.cecm_calendar = calendar_binding
        write_run_config(args, hparams=hparams)
    model = model.to(device)

    total_param, total_megabytes, _dict = count_parameters(model)
    for k, v in _dict.items():
        print("Module:", k, "param:", v, "%3.3fM" % (v / (1024 * 1024)))
    print("Total megabytes:", total_megabytes, "M")
    print("Total parameters:", total_param)

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    log_line(args, 'Number of model parameters is {}'.format(nParams))
    log_line(args, 'run_tag {} | model {} | horizon {} | seed {} | checkpoint_policy {} | split_policy {} | evaluation_mode {} | metric_protocol {} | result_dir {} | model_dir {}'.format(
        args.run_tag, args.model, args.horizon, args.seed, args.checkpoint_policy, args.split_policy,
        args.evaluation_mode, args.metric_protocol, args.result_dir, args.model_dir))
    log_line(args, 'physics_constraint disabled | combined_mmbtu_input false | lambda_conservation 0.0')
    log_line(args, 'split_counts train {} | validation {} | test_available {} | test_materialized {}'.format(
        Data.split_metadata['sample_counts']['train'], Data.split_metadata['sample_counts']['validation'],
        Data.split_metadata['sample_counts']['test_available'], Data.split_metadata['test_materialized']))
    log_line(args, 'trial_id {} | config_hash {} | cfc_hidden_size {} | period_set {} | actual_lrs single={} cfc={} pmd={} cecm={} quantile={}'.format(
        args.trial_id, args.config_hash, args.cfc_hidden_size, args.period_set, args.actual_lr_single,
        args.actual_lr_cfc, args.actual_lr_pmd, args.actual_lr_cecm, args.actual_lr_quantile))
    if args.model == 'cfc_pmd':
        log_line(args, 'cfc_pmd backbone_source {} | freeze_backbone_epochs {} | pmd_gate_init {} | pmd_head_type {} | pmd_fine_mode {} | pmd_fine_rank {}'.format(
            model.backbone_source, args.freeze_backbone_epochs, args.pmd_gate_init, args.pmd_head_type,
            args.pmd_fine_mode, args.pmd_fine_rank))
    if args.model == 'cfc_pmd_cecm':
        log_line(args, 'cfc_pmd_cecm pmd_source {} | cecm_source {} | cecm_gate_init {} | cecm_offdiag_only {} | cecm_pmd_train_start_epoch {} | cecm_cfc_train_start_epoch {} | pmd_fine_mode {} | pmd_fine_rank {} | lambda_conservation {}'.format(
            model.pmd_source, getattr(model, 'cecm_source', 'random_init'), args.cecm_gate_init,
            args.cecm_offdiag_only, args.cecm_pmd_train_start_epoch, args.cecm_cfc_train_start_epoch,
            args.pmd_fine_mode, args.pmd_fine_rank, args.lambda_conservation))
    if args.model == 'cfc_pmd_cecm_quantile':
        log_line(args, 'cfc_pmd_cecm_quantile point_source {} | quantiles {} | quantile_lr {} | spread_init {} | hidden_dim {} | fine_rank {}'.format(
            model.point_source, args.quantiles, args.quantile_lr, args.quantile_spread_init,
            args.quantile_hidden_dim, args.quantile_fine_rank))

    print('Number of model parameters is', nParams, flush=True)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).to(device)
    else:
        criterion = nn.MSELoss(size_average=False).to(device)
    evaluateL2 = nn.MSELoss(size_average=False).to(device)
    evaluateL1 = nn.L1Loss(size_average=False).to(device)

    best_val = 10000000
    best_epoch = None
    best_path = args.save if args.checkpoint_policy == 'legacy' else os.path.join(args.model_dir, 'best.pt')
    last_path = args.save if args.checkpoint_policy in ['legacy', 'last'] else os.path.join(args.model_dir, 'last.pt')
    evaluated_checkpoint = last_path if args.checkpoint_policy in ['legacy', 'last'] else best_path

    steps_per_epoch = int(np.ceil(Data.train[0].size(0) / args.batch_size))
    optim = build_optimizer(args, model, steps_per_epoch)
    legacy_mape_order = args.metric_protocol == 'legacy' and args.model == 'cfc'
    validation_history = []
    stale_epochs = 0
    optimizer_steps_completed = 0
    budget_reached = False

    def trainable_stage(epoch_number):
        """Return an integer id for the set of parameters trainable at an epoch.

        Stage ids only ever need to be compared for equality.  ``cfc``,
        ``cfc_pmd_cecm_quantile`` and the standalone baselines have a single
        stage, so they keep the historical behaviour of this loop.
        """
        if args.model == 'cfc_pmd':
            return int(epoch_number > args.freeze_backbone_epochs)
        if args.model == 'cfc_pmd_cecm':
            pmd_on = epoch_number >= args.cecm_pmd_train_start_epoch
            cfc_on = args.cecm_cfc_train_start_epoch > 0 and epoch_number >= args.cecm_cfc_train_start_epoch
            return (1 if pmd_on else 0) + (2 if cfc_on else 0)
        return 0

    current_stage = trainable_stage(1)
    stage_start_epoch = 1
    stage_count = 1

    if args.epochs == 0:
        # Initialization-only replay: validation-only mode requires best.pt,
        # but no epoch can normally create it. Preserve the initialized model
        # explicitly without running an optimizer update.
        save_model(model, best_path)
        best_epoch = 0
        log_line(args, 'initialization replay | optimizer updates 0 | checkpoint {}'.format(best_path))

    try:
        print('begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_stage = trainable_stage(epoch)
            if epoch_stage != current_stage:
                # The trainable parameter set changed, so the current patience
                # window belongs to a stage that no longer exists.  Without this
                # reset a frozen-then-unfrozen parent chain can be stopped before
                # the unfrozen stage is ever allowed to converge.
                current_stage = epoch_stage
                stage_start_epoch = epoch
                stage_count += 1
                stale_epochs = 0
            if args.model == 'cfc_pmd':
                model.set_backbone_trainable(epoch > args.freeze_backbone_epochs)
            if args.model == 'cfc_pmd_cecm':
                pmd_trainable = epoch >= args.cecm_pmd_train_start_epoch
                cfc_trainable = args.cecm_cfc_train_start_epoch > 0 and epoch >= args.cecm_cfc_train_start_epoch
                model.set_train_stage(pmd_trainable=pmd_trainable, cfc_trainable=cfc_trainable)
            if args.model == 'cfc_pmd_cecm_quantile':
                model.set_point_trainable(False)
            epoch_start_time = time.time()
            optimizer_group_lrs_used = [float(group['lr']) for group in optim.optimizer.param_groups]
            train_loss, train_mae_loss, train_data_loss, train_cons_loss = train(
                Data, Data.train[0], Data.train[1][:, :, TARGET_LOAD_COLS], model, criterion, optim,
                args.batch_size, args.horizon, args.out_nodes, lambda_conservation=args.lambda_conservation,
                label_mask=Data.train_label_mask,
                max_optimizer_updates=args.max_optimizer_updates,
                point_training_recipe=args.point_training_recipe,
                target_tau=None if a3_target_tau is None else a3_target_tau['tau'],
            )
            optimizer_steps_completed = int(getattr(optim, 'step_count', 0))
            val_pinball = None
            if args.model == 'cfc_pmd_cecm_quantile':
                val_mae, val_mape, val_corr, val_pinball = evaluate_quantile(
                    Data, Data.valid[0], Data.valid[1][:, :, :args.out_nodes], model,
                    args.batch_size, args.horizon, args.out_nodes, label_mask=Data.valid_label_mask
                )
                val_select = val_pinball
            else:
                val_mae, val_mape, val_corr = evaluate(
                    Data, Data.valid[0], Data.valid[1][:, :, :args.out_nodes], model, evaluateL2,
                    evaluateL1, args.batch_size, args.horizon, args.out_nodes, legacy_mape_order=legacy_mape_order,
                    label_mask=Data.valid_label_mask
                )
                val_select = val_mape

            optim.lronplateau(val_select)
            train_mae_losses.append(train_mae_loss)
            val_mae_losses.append(val_mae)

            epoch_seconds = time.time() - epoch_start_time
            epoch_msg = '| end of epoch {:3d} | time: {:5.2f}s | train_mape_loss {:5.4f} | train_mae_loss {:5.4f} | valid mae {:5.4f} | valid mape {:5.4f} | valid corr  {:5.4f}  learning rate  {:f}'.format(
                epoch, epoch_seconds, train_loss, train_mae_loss, val_mae, val_mape, val_corr,
                optim.optimizer.param_groups[0]['lr'])
            if val_pinball is not None:
                epoch_msg += ' | valid_pinball {:.6f}'.format(val_pinball)
            epoch_msg += ' | loss_data {:.6f} | loss_cons {:.6f} | lambda_cons {:.1e} | lambda_cons_loss {:.8f}'.format(
                train_data_loss, train_cons_loss, args.lambda_conservation, args.lambda_conservation * train_cons_loss)
            if args.model == 'cfc_pmd':
                stats = model.last_pmd_stats
                epoch_msg += ' | pmd_gate {residual_gate:.6f} | pmd_daily_abs {daily_abs_mean:.6f} | pmd_fine_std {fine_std:.6f} | backbone_trainable {}'.format(
                    epoch > args.freeze_backbone_epochs, **stats)
            if args.model == 'cfc_pmd_cecm':
                pmd_stats = model.last_pmd_stats
                cecm_stats = model.last_cecm_stats
                pmd_trainable = epoch >= args.cecm_pmd_train_start_epoch
                cfc_trainable = args.cecm_cfc_train_start_epoch > 0 and epoch >= args.cecm_cfc_train_start_epoch
                epoch_msg += ' | pmd_gate {residual_gate:.6f} | pmd_daily_abs {daily_abs_mean:.6f} | pmd_fine_std {fine_std:.6f}'.format(**pmd_stats)
                epoch_msg += ' | cecm_gate {cecm_gate:.6f} | cecm_diag_abs {diag_abs_mean:.6f} | cecm_offdiag_abs {offdiag_abs_mean:.6f} | cecm_resid_std {cross_residual_std:.6f} | pmd_trainable {} | cfc_trainable {}'.format(
                    pmd_trainable, cfc_trainable, **cecm_stats)
            if args.model == 'cfc_pmd_cecm_quantile':
                q_stats = model.last_quantile_stats
                epoch_msg += ' | q_spread_mean {spread_mean:.6f} | q_spread_min {spread_min:.6f} | q_spread_max {spread_max:.6f} | point_trainable False'.format(**q_stats)
            log_line(args, epoch_msg)
            print(epoch_msg, flush=True)

            improved = epoch >= args.best_epoch_min_offset and val_select < best_val
            if improved:
                save_model(model, best_path)
                best_val = val_select
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1

            validation_history.append({
                'epoch': epoch,
                'epoch_seconds': epoch_seconds,
                'train_mape_loss': float(train_loss),
                'train_mae_loss': float(train_mae_loss),
                'validation_mae': float(val_mae),
                'validation_mape': float(val_mape),
                'validation_corr': float(val_corr),
                'validation_pinball': None if val_pinball is None else float(val_pinball),
                'selection_metric': float(val_select),
                'improved': bool(improved),
                'stale_epochs': stale_epochs,
                'trainable_stage': epoch_stage,
                'stage_start_epoch': stage_start_epoch,
                'stage_stale_epochs': epoch - stage_start_epoch,
                'optimizer_group_lrs_used': optimizer_group_lrs_used,
                'optimizer_group_lrs': [float(group['lr']) for group in optim.optimizer.param_groups],
            })
            write_validation_history(args, validation_history)
            if args.early_stopping and stale_epochs >= args.patience:
                stop_msg = 'early stopping at epoch {} after {} validation non-improvements'.format(epoch, stale_epochs)
                log_line(args, stop_msg)
                print(stop_msg, flush=True)
                break
            if optimizer_steps_completed >= args.max_optimizer_updates:
                budget_reached = True
                budget_msg = 'optimizer update budget reached at {} steps'.format(optimizer_steps_completed)
                log_line(args, budget_msg)
                print(budget_msg, flush=True)
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    save_model(model, last_path)
    write_epoch_metrics_csv(args, validation_history)
    training_wall_seconds = time.perf_counter() - training_start_time

    model = load_model(evaluated_checkpoint)
    model = model.to(device)
    checkpoint_metric = 'pinball' if args.model == 'cfc_pmd_cecm_quantile' else 'mape'
    checkpoint_msg = "best_epoch {} | best_val_{} {:5.4f} | evaluated_checkpoint {}".format(
        best_epoch, checkpoint_metric, best_val, evaluated_checkpoint)
    log_line(args, checkpoint_msg)
    print(checkpoint_msg)

    if args.evaluation_mode == 'validation-only':
        export_split = 'validation'
        export_prefix = 'val'
        export_data = Data.valid
    else:
        if Data.test is None:
            raise RuntimeError('final-test evaluation requested without materialized test tensors')
        export_split = 'test'
        export_prefix = 'all'
        export_data = Data.test

    need_cecm_diagnostics = (
        args.model == 'cfc_pmd_cecm'
        or (args.model == 'cfc_pmd_cecm_quantile' and args.selected_point_model == 'p2b')
    )
    plow_result = plow(
        Data, export_data[0], export_data[1][:, :, :args.out_nodes], model,
        args.batch_size, args.horizon, args.out_nodes,
        label_mask=Data.valid_label_mask if export_split == 'validation' else Data.test_label_mask,
        return_mask=True,
        return_cecm_correction=need_cecm_diagnostics,
    )
    if need_cecm_diagnostics:
        all_y_true, all_predict_value, all_quantile_value, all_label_mask, cecm_correction = plow_result
    else:
        all_y_true, all_predict_value, all_quantile_value, all_label_mask = plow_result
        cecm_correction = None
    if cecm_correction is not None:
        np.save(os.path.join(args.result_dir, f'{export_prefix}_cecm_correction.npy'), cecm_correction.detach().cpu().numpy().astype(np.float32))
    if need_cecm_diagnostics:
        # Export a post-fit train-origin audit solely for seasonal coverage.
        # It is never used for checkpoint/HPO selection or performance claims.
        _, _, _, _, train_cecm_correction = plow(
            Data, Data.train[0], Data.train[1][:, :, :args.out_nodes], model,
            args.batch_size, args.horizon, args.out_nodes,
            label_mask=Data.train_label_mask,
            return_mask=True,
            return_cecm_correction=True,
        )
        if train_cecm_correction is None:
            raise RuntimeError('CECM train-origin audit did not produce a correction tensor')
        np.save(os.path.join(args.result_dir, 'train_cecm_correction.npy'), train_cecm_correction.detach().cpu().numpy().astype(np.float32))
        write_forecast_origins(
            os.path.join(args.result_dir, 'train_forecast_origins.csv'),
            Data,
            'train',
            os.path.join(os.path.dirname(os.path.abspath(args.data)), 'preprocessing_metadata.json'),
        )
    write_cecm_parameter_snapshot(model, os.path.join(args.result_dir, 'cecm_parameters.json'), args=args)
    write_forecast_origins(
        os.path.join(args.result_dir, 'forecast_origins.csv'),
        Data,
        export_split,
        os.path.join(os.path.dirname(os.path.abspath(args.data)), 'preprocessing_metadata.json'),
    )
    metrics_full = save_prediction_bundle(
        args, model, export_prefix, export_split,
        all_y_true, all_predict_value, all_quantile_value, all_label_mask
    )
    write_prediction_distribution(args, all_predict_value, all_label_mask)
    metrics_full['checkpoint_selection'] = {
        'epoch': best_epoch,
        'metric': checkpoint_metric,
        'value': float(best_val),
        'metric_protocol': args.metric_protocol,
        'checkpoint': evaluated_checkpoint,
    }
    metrics_name = 'metrics_full.json' if export_prefix == 'all' else 'val_metrics_full.json'
    with open(os.path.join(args.result_dir, metrics_name), 'w', encoding='utf-8') as handle:
        json.dump(metrics_full, handle, indent=2)
        handle.write('\n')

    overall = metrics_full['overall']
    final_msg = "final {} mae {:5.4f} | {} mape {:5.4f} | {} corr {:5.4f}".format(
        export_split, overall['mae'], export_split, overall['mape'], export_split, overall['corr'])
    log_line(args, final_msg)
    print(final_msg)
    if args.model == 'cfc_pmd_cecm':
        if getattr(model, 'cecm_conditioning', 'static') == 'linear_lead_season':
            log_line(args, 'cecm_conditional_coefficients {}'.format(json.dumps(model.cecm_parameter_snapshot(), sort_keys=True)))
        else:
            with torch.no_grad():
                delta = model.effective_delta().detach().cpu().numpy()
            log_line(args, 'cecm_delta_matrix {}'.format(np.array2string(delta, precision=8, suppress_small=False)))
        stats = model.last_cecm_stats
        log_line(args, 'cecm_final gate {cecm_gate:.6f} | diag_abs {diag_abs_mean:.6f} | offdiag_abs {offdiag_abs_mean:.6f} | cross_residual_std {cross_residual_std:.6f}'.format(**stats))
    if args.evaluation_mode == 'final-test':
        show_pred(
            all_y_true.cpu().numpy(), all_predict_value.cpu().numpy(),
            args.horizon, result_dir=args.result_dir
        )
    run_summary = {
        'trial_id': args.trial_id,
        'config_hash': args.config_hash,
        'status': 'completed',
        'model': args.model,
        'horizon': args.horizon,
        'seed': args.seed,
        'evaluation_mode': args.evaluation_mode,
        'point_training_recipe': args.point_training_recipe,
        'target_scaler_sha256': getattr(args, 'target_scaler_sha256', None),
        'target_tau_sha256': getattr(args, 'target_tau_sha256', None),
        'a3_protocol_sha256': getattr(args, 'a3_protocol_sha256', None),
        'test_materialized': Data.test is not None,
        'test_accessed': args.evaluation_mode == 'final-test',
        'epochs_completed': len(validation_history),
        'optimizer_steps_completed': int(optimizer_steps_completed),
        'max_optimizer_updates': int(args.max_optimizer_updates),
        'optimizer_update_budget': int(args.max_optimizer_updates),
        'optimizer_budget_reached': bool(budget_reached or optimizer_steps_completed >= args.max_optimizer_updates),
        'best_epoch': best_epoch,
        'best_selection_metric': float(best_val),
        'selection_metric_name': checkpoint_metric,
        'best_epoch_min_offset': int(args.best_epoch_min_offset),
        'trainable_stage_count': int(stage_count),
        'final_trainable_stage': int(current_stage),
        'checkpoint_trainable_stage': None if best_epoch is None else int(trainable_stage(best_epoch)),
        'stage_boundary_epochs': [
            int(row['epoch']) for row in validation_history
            if int(row['stage_start_epoch']) == int(row['epoch'])
        ],
        'validation_or_test_metrics': overall,
        'model_parameters': nParams,
        'actual_learning_rates': {
            'single': args.actual_lr_single,
            'cfc': args.actual_lr_cfc,
            'pmd': args.actual_lr_pmd,
            'cecm': args.actual_lr_cecm,
            'quantile': args.actual_lr_quantile,
        },
        'scheduler': {
            'lr_decay': float(args.lr_d),
            'patience': int(args.scheduler_patience),
            'early_stopping_patience': int(args.patience),
            'horizon_preset_enabled': bool(args.use_horizon_preset),
            'horizon_preset_applied': bool(args.horizon_preset_applied),
            'training_control_source': args.training_control_source,
            'best_epoch_min_offset': int(args.best_epoch_min_offset),
            'stage_aware_early_stopping': True,
        },
        'wall_seconds': time.time() - run_start_time,
        'training_wall_seconds': training_wall_seconds,
        'inference_wall_seconds': None,
        'inference_time_definition': 'not used for latency reporting; use benchmark_new_dataset_latency.py with fixed warm-up and timed blocks',
        'training_time_definition': 'process wall-clock from runner entry through final checkpoint save; includes setup, data loading, validation, and checkpoint serialization; excludes HPO orchestration and locked-test evaluation',
        'training_time_scope': 'cold_start_or_child_only',
        'hpo_included': False,
        'locked_test_included': False,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' and torch.cuda.is_available() else None,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' and torch.cuda.is_available() else None,
        'device': str(device),
    }
    with open(os.path.join(args.result_dir, 'run_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(run_summary, handle, indent=2)
        handle.write('\n')
    return overall['mae'], overall['mape'], overall['corr']


def plow(data, X, Y, model, batch_size, horizon, out_nodes, label_mask=None, return_mask=False, return_cecm_correction=False):
    model.eval()
    predictions = []
    quantile_predictions = []
    targets = []
    masks = []
    cecm_corrections = []
    batches = data.get_batches(X, Y, batch_size, False, masks=label_mask)
    for batch in batches:
        if label_mask is None:
            X, Y = batch
            batch_mask = None
        else:
            X, Y, batch_mask = batch
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
            if return_cecm_correction:
                cecm_model = get_cecm_module(model)
                correction = getattr(cecm_model, 'last_cecm_correction', None) if cecm_model is not None else None
                if correction is None:
                    cecm_corrections.append(None)
                else:
                    # CECM operates in normalized load space; only the scale
                    # changes when converting the additive correction to the
                    # physical units used by the reported diagnostics.
                    cecm_corrections.append(data._de_z_score_normalized(correction, 'gpu') - data._de_z_score_normalized(torch.zeros_like(correction), 'gpu'))
        if output.dim() == 4:
            quantile_output = output_to_quantile_shape(output, Y.size(0), horizon, out_nodes)
            output = quantile_output[..., model.median_index]
            quantile_predictions.append(denorm_load_quantiles(data, quantile_output, 'gpu'))
        else:
            output = output_to_load_shape(output, Y.size(0), horizon, out_nodes)
        y_true = data._de_z_score_normalized(Y, 'gpu')
        predict_value = data._de_z_score_normalized(output, 'gpu')
        predictions.append(predict_value)
        targets.append(y_true)
        if batch_mask is not None:
            masks.append(batch_mask)

    quantiles = torch.cat(quantile_predictions, dim=0) if quantile_predictions else None
    result = (torch.cat(targets, dim=0), torch.cat(predictions, dim=0), quantiles)
    if return_mask:
        result = result + (torch.cat(masks, dim=0) if masks else None,)
    if return_cecm_correction:
        if cecm_corrections and all(item is not None for item in cecm_corrections):
            result = result + (torch.cat(cecm_corrections, dim=0),)
        else:
            result = result + (None,)
    return result


if __name__ == "__main__":
    main()
