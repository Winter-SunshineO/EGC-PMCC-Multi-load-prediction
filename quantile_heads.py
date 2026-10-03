import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value):
    value = float(value)
    return math.log(math.exp(value) - 1.0)


def _zero_init_last_linear(module, bias_value=None):
    last_linear = None
    if isinstance(module, nn.Linear):
        last_linear = module
    else:
        for child in module.modules():
            if isinstance(child, nn.Linear):
                last_linear = child
    if last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            if bias_value is None:
                nn.init.zeros_(last_linear.bias)
            else:
                nn.init.constant_(last_linear.bias, float(bias_value))


def _make_head(context_dim, out_dim, hidden_dim, dropout, bias_value=None):
    head = nn.Sequential(
        nn.Linear(context_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )
    _zero_init_last_linear(head, bias_value=bias_value)
    return head


def _make_intraday_basis(rank):
    hours = torch.arange(24, dtype=torch.float32)
    basis = []
    harmonic = 1
    while len(basis) < rank:
        angle = 2.0 * math.pi * harmonic * hours / 24.0
        basis.append(torch.sin(angle))
        if len(basis) < rank:
            basis.append(torch.cos(angle))
        harmonic += 1

    basis = torch.stack(basis[:rank], dim=0)
    basis = basis - basis.mean(dim=1, keepdim=True)
    scale = basis.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return basis / scale


class CfcPMDCECMQuantile(nn.Module):
    """Calibration-only quantile wrapper around a frozen point backbone.

    The point model remains responsible for q50. The wrapper learns positive
    spreads around that median, so point forecasts are not changed by P4.
    Input: [B, 1, C, L]. Output: [B, H, c_out, Q].
    """

    def __init__(
        self,
        point_model,
        pred_len,
        c_out=3,
        quantiles=(0.1, 0.5, 0.9),
        hidden_dim=32,
        dropout=0.1,
        fine_rank=2,
        spread_init=0.2,
    ):
        super().__init__()
        quantiles = sorted(float(q) for q in quantiles)
        if 0.5 not in quantiles:
            raise ValueError(f"quantiles must include 0.5, got {quantiles}")
        if any(q <= 0.0 or q >= 1.0 for q in quantiles):
            raise ValueError(f"quantiles must be inside (0, 1), got {quantiles}")
        if len(set(quantiles)) != len(quantiles):
            raise ValueError(f"quantiles must be unique, got {quantiles}")

        self.point_model = point_model
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self.days = self.pred_len // 24
        if self.pred_len % 24 != 0:
            raise ValueError(f"quantile wrapper expects pred_len divisible by 24, got {self.pred_len}")
        self.fine_rank = int(fine_rank)
        if self.fine_rank <= 0:
            raise ValueError(f"fine_rank must be positive, got {self.fine_rank}")

        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.median_index = quantiles.index(0.5)
        self.nonmedian_indices = [idx for idx, q in enumerate(quantiles) if q != 0.5]
        self.n_spreads = len(self.nonmedian_indices)
        if self.n_spreads == 0:
            raise ValueError("at least one non-median quantile is required")

        self.context_dim = 15 + self.days * self.c_out
        bias_value = _inverse_softplus(spread_init)
        self.daily_head = _make_head(
            self.context_dim,
            self.days * self.c_out * self.n_spreads,
            hidden_dim,
            dropout,
            bias_value=bias_value,
        )
        self.fine_head = _make_head(
            self.context_dim,
            self.days * self.c_out * self.n_spreads * self.fine_rank,
            hidden_dim,
            dropout,
            bias_value=0.0,
        )
        self.fine_basis = nn.Parameter(_make_intraday_basis(self.fine_rank))
        self.freeze_point = True
        self.point_source = "random_init"
        self.last_quantile_stats = {}
        self.set_point_trainable(False)

    def set_point_trainable(self, trainable):
        self.freeze_point = not trainable
        for param in self.point_model.parameters():
            param.requires_grad_(trainable)
        if self.freeze_point:
            self.point_model.eval()
        else:
            self.point_model.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_point:
            self.point_model.eval()
        return self

    def quantile_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("point_model."):
                yield param

    def _context(self, x, point):
        x0 = x.squeeze(1).permute(0, 2, 1)[:, :, : self.c_out]
        recent = x0[:, -24:, :]
        pieces = [
            x0[:, -1, :],
            recent.mean(dim=1),
            torch.sqrt(torch.var(recent, dim=1, unbiased=False) + 1e-5),
            x0.mean(dim=1),
            torch.sqrt(torch.var(x0, dim=1, unbiased=False) + 1e-5),
            point.reshape(point.size(0), self.days, 24, self.c_out).mean(dim=2).reshape(point.size(0), -1),
        ]
        return torch.cat(pieces, dim=1)

    def forward(self, x):
        if self.freeze_point:
            with torch.no_grad():
                point = self.point_model(x)
        else:
            point = self.point_model(x)

        context = self._context(x, point.detach() if self.freeze_point else point)
        daily = self.daily_head(context).reshape(-1, self.days, self.c_out, self.n_spreads)
        daily = daily.unsqueeze(2).repeat(1, 1, 24, 1, 1)

        coeff = self.fine_head(context).reshape(-1, self.days, self.c_out, self.n_spreads, self.fine_rank)
        fine = torch.einsum("bdcqr,rh->bdhcq", coeff, self.fine_basis)
        fine = fine - fine.mean(dim=2, keepdim=True)

        spread = F.softplus(daily + fine).reshape(-1, self.pred_len, self.c_out, self.n_spreads)
        outputs = []
        spread_pos = 0
        quantiles = self.quantiles.detach().cpu().tolist()
        for idx, quantile in enumerate(quantiles):
            if idx == self.median_index:
                outputs.append(point)
                continue
            this_spread = spread[:, :, :, spread_pos]
            spread_pos += 1
            if quantile < 0.5:
                outputs.append(point - this_spread)
            else:
                outputs.append(point + this_spread)
        out = torch.stack(outputs, dim=-1)

        with torch.no_grad():
            self.last_quantile_stats = {
                "spread_mean": float(spread.detach().mean().cpu().item()),
                "spread_min": float(spread.detach().min().cpu().item()),
                "spread_max": float(spread.detach().max().cpu().item()),
                "spread_std": float(spread.detach().std(unbiased=False).cpu().item()),
            }
        return out

    def quantile_metadata(self):
        return {
            "quantiles": [float(q) for q in self.quantiles.detach().cpu().tolist()],
            "median_index": int(self.median_index),
            "point_source": self.point_source,
        }

    def write_quantile_metadata(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.quantile_metadata(), handle, indent=2)
            handle.write("\n")
