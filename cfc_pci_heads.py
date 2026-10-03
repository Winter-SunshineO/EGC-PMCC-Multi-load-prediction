import math

import torch
import torch.nn as nn


def _zero_init_last_linear(module):
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
            nn.init.zeros_(last_linear.bias)


def _make_head(context_dim, out_dim, head_type, hidden_dim, dropout):
    if head_type == "linear":
        head = nn.Linear(context_dim, out_dim)
    elif head_type == "mlp":
        head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    else:
        raise ValueError(f"unknown PMD head type {head_type}")
    _zero_init_last_linear(head)
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


class CfcWithPMD(nn.Module):
    """Cfc backbone with a small PMD residual correction head.

    The wrapped Cfc remains responsible for the base point forecast. PMD only
    predicts a gated residual with daily level plus zero-mean intraday shape.
    Input: [B, 1, C, L]. Output: [B, H, c_out].
    """

    def __init__(
        self,
        backbone,
        pred_len,
        c_out=3,
        gate_init=-3.0,
        head_type="linear",
        hidden_dim=64,
        dropout=0.1,
        fine_mode="direct",
        fine_rank=2,
    ):
        super().__init__()
        self.backbone = backbone
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self.fine_mode = fine_mode
        self.fine_rank = int(fine_rank)
        self.days = self.pred_len // 24
        if self.pred_len % 24 != 0:
            raise ValueError(f"PMD expects pred_len divisible by 24, got {self.pred_len}")
        if self.fine_mode not in ["direct", "lowrank"]:
            raise ValueError(f"unknown PMD fine mode {self.fine_mode}")
        if self.fine_rank <= 0:
            raise ValueError(f"fine_rank must be positive, got {self.fine_rank}")

        self.context_dim = 15 + self.days * self.c_out
        self.daily_head = _make_head(
            self.context_dim,
            self.days * self.c_out,
            head_type,
            hidden_dim,
            dropout,
        )
        if self.fine_mode == "direct":
            self.fine_head = _make_head(
                self.context_dim,
                self.pred_len * self.c_out,
                head_type,
                hidden_dim,
                dropout,
            )
            self.fine_basis = None
        else:
            self.fine_head = _make_head(
                self.context_dim,
                self.days * self.c_out * self.fine_rank,
                head_type,
                hidden_dim,
                dropout,
            )
            self.fine_basis = nn.Parameter(_make_intraday_basis(self.fine_rank))
        self.pmd_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.freeze_backbone = False
        self.backbone_source = "random_init"
        self.last_pmd_stats = {}

    def backbone_parameters(self):
        return self.backbone.parameters()

    def pmd_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("backbone."):
                yield param

    def set_backbone_trainable(self, trainable):
        self.freeze_backbone = not trainable
        for param in self.backbone.parameters():
            param.requires_grad_(trainable)
        if self.freeze_backbone:
            self.backbone.eval()
        else:
            self.backbone.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _context(self, x, base):
        x0 = x.squeeze(1).permute(0, 2, 1)[:, :, : self.c_out]
        recent = x0[:, -24:, :]
        pieces = [
            x0[:, -1, :],
            recent.mean(dim=1),
            torch.sqrt(torch.var(recent, dim=1, unbiased=False) + 1e-5),
            x0.mean(dim=1),
            torch.sqrt(torch.var(x0, dim=1, unbiased=False) + 1e-5),
            base.reshape(base.size(0), self.days, 24, self.c_out).mean(dim=2).reshape(base.size(0), -1),
        ]
        return torch.cat(pieces, dim=1)

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                base = self.backbone(x)
        else:
            base = self.backbone(x)

        context = self._context(x, base.detach() if self.freeze_backbone else base)
        daily = self.daily_head(context).reshape(-1, self.days, self.c_out)
        daily_full = daily.repeat_interleave(24, dim=1)

        if self.fine_mode == "direct":
            fine = self.fine_head(context).reshape(-1, self.days, 24, self.c_out)
        else:
            coeff = self.fine_head(context).reshape(-1, self.days, self.c_out, self.fine_rank)
            fine = torch.einsum("bdcr,rh->bdhc", coeff, self.fine_basis)
        fine = fine - fine.mean(dim=2, keepdim=True)
        fine = fine.reshape(-1, self.pred_len, self.c_out)

        residual = daily_full + fine
        gate = torch.sigmoid(self.pmd_gate)
        out = base + gate * residual

        with torch.no_grad():
            self.last_pmd_stats = {
                "residual_gate": float(gate.detach().cpu().item()),
                "daily_mean": float(daily_full.detach().mean().cpu().item()),
                "daily_abs_mean": float(daily_full.detach().abs().mean().cpu().item()),
                "fine_std": float(fine.detach().std(unbiased=False).cpu().item()),
                "residual_std": float(residual.detach().std(unbiased=False).cpu().item()),
            }
        return out


class CfcWithPMDCECM(nn.Module):
    """Gated near-identity cross-energy correction on top of CfcWithPMD."""

    def __init__(self, pmd_model, c_out=3, gate_init=-3.0, offdiag_only=False,
                 cecm_trainable=True, conditioning="static"):
        super().__init__()
        self.pmd = pmd_model
        self.c_out = int(c_out)
        self.offdiag_only = bool(offdiag_only)
        if conditioning not in {"static", "linear_lead", "linear_lead_season"}:
            raise ValueError(f"unknown CECM conditioning {conditioning}")
        self.cecm_conditioning = conditioning
        self.cecm_delta = nn.Parameter(torch.zeros(self.c_out, self.c_out))
        self.cecm_gate = nn.Parameter(torch.tensor(float(gate_init)))
        if conditioning in {"linear_lead", "linear_lead_season"}:
            if self.pmd.pred_len < 2:
                raise ValueError("linear_lead requires at least two forecast steps")
            self.cecm_delta_slope = nn.Parameter(torch.zeros(self.c_out, self.c_out))
            self.register_buffer("cecm_lead_coordinate", torch.linspace(-1, 1, self.pmd.pred_len))
        if conditioning == "linear_lead_season":
            self.cecm_delta_sin = nn.Parameter(torch.zeros(self.c_out, self.c_out))
            self.cecm_delta_cos = nn.Parameter(torch.zeros(self.c_out, self.c_out))
            self.register_buffer("cecm_calendar_indices", torch.full((2,), -1, dtype=torch.long))
            self.register_buffer("cecm_calendar_mean", torch.zeros(2))
            self.register_buffer("cecm_calendar_scale", torch.ones(2))
            self.register_buffer("cecm_calendar_ready", torch.tensor(False))
        self.cecm_trainable = bool(cecm_trainable)
        for param in self.cecm_parameters():
            param.requires_grad_(self.cecm_trainable)
        mask = torch.ones(self.c_out, self.c_out)
        if self.offdiag_only:
            mask.fill_(1.0)
            mask.fill_diagonal_(0.0)
        self.register_buffer("cecm_mask", mask)
        self.freeze_pmd = False
        self.pmd_source = "random_init"
        self.last_pmd_stats = {}
        self.last_cecm_stats = {}
        self.last_cecm_correction = None

    def cfc_parameters(self):
        return self.pmd.backbone_parameters()

    def pmd_parameters(self):
        return self.pmd.pmd_parameters()

    def cecm_parameters(self):
        yield self.cecm_delta
        yield self.cecm_gate
        if getattr(self, "cecm_conditioning", "static") in {"linear_lead", "linear_lead_season"}:
            yield self.cecm_delta_slope
        if getattr(self, "cecm_conditioning", "static") == "linear_lead_season":
            yield self.cecm_delta_sin
            yield self.cecm_delta_cos

    def configure_season_calendar(self, indices, mean, scale):
        if getattr(self, "cecm_conditioning", "static") != "linear_lead_season":
            raise ValueError("calendar binding requires seasonal conditioning")
        device = self.cecm_calendar_indices.device
        raw_indices = torch.as_tensor(indices, device=device)
        indices = raw_indices.to(torch.long)
        mean = torch.as_tensor(mean, dtype=self.cecm_calendar_mean.dtype, device=device)
        scale = torch.as_tensor(scale, dtype=self.cecm_calendar_scale.dtype, device=device)
        if (indices.shape != (2,) or mean.shape != (2,) or scale.shape != (2,)
                or not torch.equal(raw_indices, indices) or (indices < 3).any()
                or indices[0] == indices[1] or not torch.isfinite(mean).all()
                or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("invalid calendar column indices or train-only normalization")
        pairs = ((self.cecm_calendar_indices, indices), (self.cecm_calendar_mean, mean),
                 (self.cecm_calendar_scale, scale))
        if self.cecm_calendar_ready.item() and any(not torch.equal(a, b) for a, b in pairs):
            raise ValueError("refusing to rebind checkpoint calendar normalization")
        with torch.no_grad():
            for buffer, value in pairs:
                buffer.copy_(value)
            self.cecm_calendar_ready.fill_(True)

    def season_calendar(self, x):
        if (getattr(self, "cecm_conditioning", "static") != "linear_lead_season"
                or not self.cecm_calendar_ready.item()):
            raise ValueError("seasonal CECM requires a bound train-only calendar schema")
        if (x.ndim != 4 or x.shape[1] != 1 or x.shape[-1] < 1
                or (self.cecm_calendar_indices >= x.shape[2]).any()):
            raise ValueError("calendar input must have shape [batch, 1, features, history]")
        # The last historical hour is the forecast origin, not a future observation.
        normalized = x[:, 0, :, -1].index_select(1, self.cecm_calendar_indices)
        phase = normalized * self.cecm_calendar_scale + self.cecm_calendar_mean
        if not torch.isfinite(phase).all():
            raise ValueError("nonfinite origin calendar")
        return phase

    def set_train_stage(self, pmd_trainable, cfc_trainable=False):
        self.freeze_pmd = not pmd_trainable
        self.pmd.set_backbone_trainable(cfc_trainable)
        for param in self.pmd.pmd_parameters():
            param.requires_grad_(pmd_trainable)
        for param in self.cecm_parameters():
            # Old full-object checkpoints predate the ablation switch.
            param.requires_grad_(getattr(self, "cecm_trainable", True))
        if self.freeze_pmd:
            self.pmd.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_pmd:
            self.pmd.eval()
        return self

    def effective_delta(self, calendar=None):
        mode = getattr(self, "cecm_conditioning", "static")
        if mode == "linear_lead_season":
            if (not isinstance(calendar, torch.Tensor) or calendar.ndim != 2
                    or calendar.shape[1] != 2 or not torch.isfinite(calendar).all()):
                raise ValueError("seasonal effective_delta requires explicit finite [batch, 2] calendar")
            calendar = calendar.to(device=self.cecm_delta.device, dtype=self.cecm_delta.dtype)
            lead = (self.cecm_delta.unsqueeze(0)
                    + self.cecm_lead_coordinate[:, None, None] * self.cecm_delta_slope.unsqueeze(0))
            return (lead.unsqueeze(0)
                    + calendar[:, 0, None, None, None] * self.cecm_delta_sin
                    + calendar[:, 1, None, None, None] * self.cecm_delta_cos) * self.cecm_mask
        if mode == "linear_lead":
            return (self.cecm_delta.unsqueeze(0)
                    + self.cecm_lead_coordinate[:, None, None] * self.cecm_delta_slope.unsqueeze(0)) * self.cecm_mask
        return self.cecm_delta * self.cecm_mask

    def cecm_parameter_snapshot(self):
        """Export coefficients, not a fictitious static matrix for seasonal models."""
        gate_logit = float(self.cecm_gate.detach().cpu().item())
        gate = float(torch.sigmoid(self.cecm_gate).detach().cpu().item())
        if getattr(self, "cecm_conditioning", "static") == "linear_lead_season":
            return {
                "parameterization": "z'_h = z_h + sigmoid(g) * (Delta0 + q_h*Delta1 + s_origin*DeltaS + c_origin*DeltaC) @ z_h",
                "is_static_matrix": False,
                "lead_conditioning": "linear_lead",
                "season_conditioning": "origin_day_of_year_sin_cos",
                "applied_effect_analysis": "correction_tensor_grouped_by_origin_season",
                "c_out": int(self.c_out),
                "offdiag_only": bool(self.offdiag_only),
                "cecm_gate_logit": gate_logit,
                "cecm_gate": gate,
                "lead_coordinate": self.cecm_lead_coordinate.detach().cpu().tolist(),
                "delta_intercept_matrix": (self.cecm_delta * self.cecm_mask).detach().cpu().tolist(),
                "delta_slope_matrix": (self.cecm_delta_slope * self.cecm_mask).detach().cpu().tolist(),
                "delta_season_sin_matrix": (self.cecm_delta_sin * self.cecm_mask).detach().cpu().tolist(),
                "delta_season_cos_matrix": (self.cecm_delta_cos * self.cecm_mask).detach().cpu().tolist(),
                "matrix_axes": ["origin", "lead", "output_load", "input_load"],
                "delta_matrix": None,
                "effective_correction_matrix": None,
                "calendar": {
                    "feature_names": ["DayOfYear_sin", "DayOfYear_cos"],
                    "column_indices": self.cecm_calendar_indices.detach().cpu().tolist(),
                    "normalization_mean": self.cecm_calendar_mean.detach().cpu().tolist(),
                    "normalization_scale": self.cecm_calendar_scale.detach().cpu().tolist(),
                    "ready": bool(self.cecm_calendar_ready.item()),
                    "source": "last_observed_input_hour",
                    "held_constant_across_forecast": True,
                },
            }
        delta = self.effective_delta().detach().cpu()
        effective = gate * delta
        if getattr(self, "cecm_conditioning", "static") == "linear_lead":
            return {
                "parameterization": "z'_h = z_h + sigmoid(g) * (Delta0 + q_h * Delta1) @ z_h",
                "is_static_matrix": False,
                "lead_conditioning": "linear_lead",
                "season_conditioning": "none",
                "applied_effect_analysis": "correction_tensor_grouped_by_origin_season",
                "c_out": int(self.c_out),
                "offdiag_only": bool(self.offdiag_only),
                "cecm_gate_logit": gate_logit,
                "cecm_gate": gate,
                "lead_coordinate": self.cecm_lead_coordinate.detach().cpu().tolist(),
                "delta_intercept_matrix": (self.cecm_delta * self.cecm_mask).detach().cpu().tolist(),
                "delta_slope_matrix": (self.cecm_delta_slope * self.cecm_mask).detach().cpu().tolist(),
                "matrix_axes": ["lead", "output_load", "input_load"],
                "delta_matrix": delta.tolist(),
                "effective_correction_matrix": effective.tolist(),
            }
        return {
            "parameterization": "Y = Y_pmd + sigmoid(gate_logit) * Y_pmd @ Delta.T",
            "is_static_matrix": True,
            "season_conditioning": "none",
            "applied_effect_analysis": "correction_tensor_grouped_by_origin_season",
            "c_out": int(self.c_out),
            "offdiag_only": bool(self.offdiag_only),
            "cecm_gate_logit": gate_logit,
            "cecm_gate": gate,
            "delta_matrix": delta.tolist(),
            "effective_correction_matrix": effective.tolist(),
        }

    def forward(self, x):
        if self.freeze_pmd:
            with torch.no_grad():
                pmd_out = self.pmd(x)
        else:
            pmd_out = self.pmd(x)

        mode = getattr(self, "cecm_conditioning", "static")
        delta = self.effective_delta(self.season_calendar(x)) if mode == "linear_lead_season" else self.effective_delta()
        if mode == "linear_lead_season":
            cross_residual = torch.einsum("bhi,bhji->bhj", pmd_out, delta)
        elif mode == "linear_lead":
            cross_residual = torch.einsum("bhi,hji->bhj", pmd_out, delta)
        else:
            cross_residual = torch.matmul(pmd_out, delta.t())
        gate = torch.sigmoid(self.cecm_gate)
        correction = gate * cross_residual
        out = pmd_out + correction

        with torch.no_grad():
            diag = torch.diagonal(delta, dim1=-2, dim2=-1)
            offdiag_mask = 1.0 - torch.eye(self.c_out, device=delta.device, dtype=delta.dtype)
            offdiag = delta * offdiag_mask
            matrix_count = math.prod(delta.shape[:-2])
            self.last_pmd_stats = dict(getattr(self.pmd, "last_pmd_stats", {}))
            self.last_cecm_correction = correction.detach()
            self.last_cecm_stats = {
                "cecm_gate": float(gate.detach().cpu().item()),
                "diag_abs_mean": float(diag.detach().abs().mean().cpu().item()),
                "offdiag_abs_mean": float(offdiag.detach().abs().sum().cpu().item() / max(1, matrix_count * self.c_out * (self.c_out - 1))),
                "cross_residual_std": float(cross_residual.detach().std(unbiased=False).cpu().item()),
                "delta_matrix": None if mode == "linear_lead_season" else delta.detach().cpu().tolist(),
                "effective_correction_matrix": None if mode == "linear_lead_season" else (gate * delta).detach().cpu().tolist(),
            }
        return out
