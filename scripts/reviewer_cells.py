"""Recurrent model and validation kernels used by the reproduction runner."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch_cfc import CfcCell, Multi_period_predication, Sequential_projection
from metrics import metric_overall
from util import DataLoaderS

VARIANTS = ("cfc", "gru", "lstm", "vanilla_rnn")

def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hparams(hidden_size: int) -> dict[str, Any]:
    return {
        "backbone_units": hidden_size,
        "backbone_dr": 0.2,
        "backbone_layers": 2,
        "backbone_activation": "lecun",
        "init": 0.53,
        "minimal": False,
        "no_gate": False,
    }


class MatchedRecurrentForecaster(nn.Module):
    """The fixed E08 wrapper; only ``cell`` and its state width vary."""

    def __init__(self, variant: str, hidden_size: int, horizon: int, period_set: list[int], common_hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.hidden_size = int(hidden_size)
        self.common_hidden = int(common_hidden if common_hidden is not None else hidden_size)
        self.horizon = int(horizon)
        self.input_size = 12
        self.output_size = 3
        self.encoder = Multi_period_predication(168, 168, self.input_size, torch.tensor(period_set, dtype=torch.int32))
        self.x_pro = Multi_period_predication(168, 168, self.input_size, torch.tensor(period_set, dtype=torch.int32))
        if variant == "cfc":
            if self.hidden_size != self.common_hidden:
                raise ValueError("CfC reference state width must equal d_common")
            self.cell = CfcCell(self.input_size, self.hidden_size, hparams(self.common_hidden))
            self.cell_state_size = self.hidden_size
            # Keep an explicit interface adapter in every variant.  This makes
            # the outer scaffold identical and makes adapter parameters part
            # of the pre-registered parameter accounting.
            self.output_adapter = nn.Linear(self.hidden_size, self.common_hidden)
        elif variant == "gru":
            self.cell = nn.GRUCell(self.input_size, self.hidden_size)
            self.cell_state_size = self.hidden_size
            self.output_adapter = nn.Linear(self.hidden_size, self.common_hidden)
        elif variant == "lstm":
            self.cell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.cell_state_size = self.hidden_size
            self.output_adapter = nn.Linear(self.hidden_size, self.common_hidden)
        else:
            self.cell = nn.RNNCell(self.input_size, self.hidden_size, nonlinearity="tanh")
            self.cell_state_size = self.hidden_size
            self.output_adapter = nn.Linear(self.hidden_size, self.common_hidden)
        # A fixed common state interface is used by the projection.  The
        # selected registry widths are the common widths for each variant.
        self.horizon_projector = Sequential_projection(24 * 7 * 2, horizon)
        self.readout = nn.Linear(self.common_hidden, self.output_size)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()

    def update_block_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.cell.parameters()) + sum(parameter.numel() for parameter in self.output_adapter.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != 168 or x.shape[2] != 12:
            raise ValueError(f"expected [B,168,12], got {tuple(x.shape)}")
        sequence_mean = torch.mean(x, dim=1, keepdim=True)
        centered = x - sequence_mean
        mean_out = sequence_mean[:, :, :3]
        projected = self.encoder(centered).permute(0, 2, 1)
        stream2 = self.x_pro(projected).permute(0, 2, 1)
        batch_size = projected.shape[0]
        # [B,168,12] -> [B,24,7,12], exactly matching torch_cfc.Cfc.
        stream_states: list[torch.Tensor] = []
        for stream, delta in ((projected, 1.0), (stream2, 1.0 / 168.0)):
            reshaped = stream.permute(0, 2, 1).reshape(batch_size, 12, 7, 24).permute(0, 3, 2, 1)
            if self.variant == "lstm":
                h = torch.zeros(batch_size, 7, self.hidden_size, device=x.device, dtype=x.dtype)
                c = torch.zeros_like(h)
            else:
                h = torch.zeros(batch_size, 7, self.hidden_size, device=x.device, dtype=x.dtype)
            outputs: list[torch.Tensor] = []
            for step in range(24):
                current = reshaped[:, step]
                if self.variant == "cfc":
                    h = self.cell(current, h, x.new_full((batch_size,), delta))
                elif self.variant == "lstm":
                    flat_h, flat_c = h.reshape(-1, self.hidden_size), c.reshape(-1, self.hidden_size)
                    flat_current = current.reshape(-1, self.input_size)
                    h, c = self.cell(flat_current, (flat_h, flat_c))
                    h = h.reshape(batch_size, 7, self.hidden_size)
                    c = c.reshape(batch_size, 7, self.hidden_size)
                else:
                    flat_h = h.reshape(-1, self.hidden_size)
                    flat_current = current.reshape(-1, self.input_size)
                    h = self.cell(flat_current, flat_h).reshape(batch_size, 7, self.hidden_size)
                outputs.append(self.output_adapter(h))
            stream_states.append(torch.stack(outputs, dim=1))
        joined = torch.cat(stream_states, dim=1)  # [B,48,7,d_common]
        # Match Cfc exactly: [B,hidden,7*48] -> Linear(336,H).
        joined = joined.permute(0, 3, 2, 1).reshape(batch_size, self.common_hidden, 7 * 48)
        hidden = self.horizon_projector.linear(joined).permute(0, 2, 1)
        output = self.readout(self.dropout(hidden))
        return output + mean_out.expand(-1, self.horizon, -1)


def _input_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch


def _denorm(data: DataLoaderS, values: torch.Tensor) -> torch.Tensor:
    std = torch.from_numpy(data.scale_std[:3]).to(values.device, values.dtype).view(1, 1, 3)
    mean = torch.from_numpy(data.scale_mean[:3]).to(values.device, values.dtype).view(1, 1, 3)
    return values * std + mean


def evaluate_validation(data: DataLoaderS, model: nn.Module, device: torch.device, batch_size: int, horizon: int) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets, masks = [], [], []
    offset = 0
    with torch.no_grad():
        for x, y, batch_mask in data.get_batches(data.valid[0], data.valid[1], batch_size, False, masks=data.valid_label_mask):
            x = _input_batch(x).to(device)
            prediction = model(x)
            predictions.append(prediction.detach().cpu())
            targets.append(y[:, :, :3].detach().cpu())
            masks.append(batch_mask.detach().cpu())
            offset += len(x)
    prediction_norm = torch.cat(predictions, dim=0).to(device)
    target_norm = torch.cat(targets, dim=0).to(device)
    mask = torch.cat(masks, dim=0).numpy().astype(bool) if masks else np.ones(target_norm.shape, dtype=bool)
    prediction = _denorm(data, prediction_norm).cpu().numpy()
    target = _denorm(data, target_norm).cpu().numpy()
    summary = metric_overall(target, prediction, mask)
    return float(summary["mape"]), float(summary["mae"]), float(summary["corr"]), target, prediction, mask
