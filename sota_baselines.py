from types import SimpleNamespace

import torch
import torch.nn as nn

from iTransformer import Model as ITransformerModel


class ITransformerBaseline(nn.Module):
    """Plain iTransformer baseline with the repository train.py IO contract.

    Input: [B, 1, C, L]. Output: [B, H, c_out].
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=12,
        c_out=3,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_ff=64,
        dropout=0.2,
        factor=1,
        activation="gelu",
    ):
        super().__init__()
        self.c_out = int(c_out)
        configs = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            enc_in=int(enc_in),
            c_out=int(c_out),
            d_model=int(d_model),
            n_heads=int(n_heads),
            e_layers=int(e_layers),
            d_ff=int(d_ff),
            dropout=float(dropout),
            factor=int(factor),
            activation=activation,
            embed="timeF",
            freq="h",
        )
        self.model = ITransformerModel(configs)

    def forward(self, x):
        x_enc = x.squeeze(1).permute(0, 2, 1)
        output = self.model(x_enc, None, None, None)
        return output[:, :, : self.c_out]


class PatchTSTBaseline(nn.Module):
    """Faithful lightweight PatchTST baseline with the train.py IO contract.

    Keeps the three defining PatchTST design choices:
    - channel independence (each variable is encoded by the same shared Transformer);
    - patching (the lookback window is split into overlapping patches that become tokens);
    - RevIN-style instance normalization per channel.

    Input: [B, 1, C, L]. Output: [B, H, c_out] (only the first c_out load channels).
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=12,
        c_out=3,
        d_model=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        dropout=0.2,
        patch_len=16,
        stride=8,
        activation="gelu",
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        # number of patches after padding the end by `stride` (PatchTST end-padding)
        self.num_patches = (self.seq_len + self.stride - self.patch_len) // self.stride + 1
        self.pad = nn.ReplicationPad1d((0, self.stride))

        self.value_embedding = nn.Linear(self.patch_len, int(d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, int(d_model)) * 0.02)
        self.dropout = nn.Dropout(float(dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(e_layers))
        self.head = nn.Linear(int(d_model) * self.num_patches, self.pred_len)

    def forward(self, x):
        # [B, 1, C, L] -> [B, C, L]
        x_enc = x.squeeze(1)
        B, C, L = x_enc.shape

        # RevIN-style per-channel instance norm
        means = x_enc.mean(dim=2, keepdim=True)
        stdev = torch.sqrt(x_enc.var(dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        # patching: [B, C, L] -> [B*C, num_patches, patch_len]
        z = self.pad(x_norm)
        z = z.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [B, C, num_patches, patch_len]
        z = z.reshape(B * C, self.num_patches, self.patch_len)

        tokens = self.value_embedding(z) + self.pos_embedding
        tokens = self.dropout(tokens)
        enc = self.encoder(tokens)  # [B*C, num_patches, d_model]

        flat = enc.reshape(B * C, -1)
        pred = self.head(flat)  # [B*C, pred_len]
        pred = pred.reshape(B, C, self.pred_len)

        # de-normalize per channel
        pred = pred * stdev + means
        pred = pred.permute(0, 2, 1)  # [B, pred_len, C]
        return pred[:, :, : self.c_out]


class _SeriesDecomp(nn.Module):
    """Moving-average series decomposition into trend and seasonal parts."""

    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):  # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class TimeMixerBaseline(nn.Module):
    """Faithful lightweight TimeMixer baseline with the train.py IO contract.

    Keeps the defining TimeMixer ideas:
    - multiscale views via average-pool downsampling;
    - per-scale season/trend decomposition (Past-Decomposable-Mixing);
    - cross-scale mixing then multiscale prediction summation (Future-Multipredictor-Mixing).

    Input: [B, 1, C, L]. Output: [B, H, c_out].
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=12,
        c_out=3,
        d_model=32,
        e_layers=2,
        dropout=0.2,
        num_scales=3,
        down_factor=2,
        decomp_kernel=25,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.num_scales = int(num_scales)
        self.down_factor = int(down_factor)

        self.scale_lens = [self.seq_len // (self.down_factor ** s) for s in range(self.num_scales)]
        self.decomp = _SeriesDecomp(decomp_kernel)
        self.dropout = nn.Dropout(float(dropout))

        # per-scale embedding into d_model then a small mixing MLP over time (PDM)
        self.embed = nn.ModuleList([nn.Linear(L, int(d_model)) for L in self.scale_lens])
        mix_layers = []
        for _ in range(int(e_layers)):
            mix_layers.append(
                nn.Sequential(
                    nn.Linear(int(d_model), int(d_model)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(d_model), int(d_model)),
                )
            )
        self.mixers = nn.ModuleList(mix_layers)
        # per-scale predictor head -> pred_len (FMM), summed across scales
        self.predictors = nn.ModuleList([nn.Linear(int(d_model), self.pred_len) for _ in range(self.num_scales)])

    def forward(self, x):
        x_enc = x.squeeze(1).permute(0, 2, 1)  # [B, L, C]
        B, L, C = x_enc.shape

        means = x_enc.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        pred_sum = x_norm.new_zeros(B, self.pred_len, C)
        current = x_norm
        for s in range(self.num_scales):
            if s > 0:
                cur_t = current.permute(0, 2, 1)
                cur_t = nn.functional.avg_pool1d(cur_t, kernel_size=self.down_factor, stride=self.down_factor)
                current = cur_t.permute(0, 2, 1)
            target_len = self.scale_lens[s]
            scale_view = current[:, :target_len, :]

            seasonal, trend = self.decomp(scale_view)
            mixed_input = (seasonal + trend).permute(0, 2, 1)  # [B, C, target_len]
            h = self.embed[s](mixed_input)  # [B, C, d_model]
            for mixer in self.mixers:
                h = h + mixer(h)
            h = self.dropout(h)
            scale_pred = self.predictors[s](h).permute(0, 2, 1)  # [B, pred_len, C]
            pred_sum = pred_sum + scale_pred

        pred = pred_sum * stdev + means
        return pred[:, :, : self.c_out]


class _InceptionBlock2D(nn.Module):
    """Small multi-kernel 2D conv block used inside a TimesBlock."""

    def __init__(self, in_ch, out_ch, num_kernels=3):
        super().__init__()
        self.kernels = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i) for i in range(num_kernels)]
        )

    def forward(self, x):
        out = 0.0
        for kernel in self.kernels:
            out = out + kernel(x)
        return out / len(self.kernels)


class TimesNetBaseline(nn.Module):
    """Faithful lightweight TimesNet baseline with the train.py IO contract.

    Keeps the defining TimesNet mechanism:
    - FFT to discover top-k dominant periods;
    - reshape the 1D sequence into 2D (period x frequency) tensors;
    - 2D Inception convolution per period;
    - amplitude-weighted aggregation across periods.

    Input: [B, 1, C, L]. Output: [B, H, c_out].
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=12,
        c_out=3,
        d_model=32,
        e_layers=2,
        d_ff=32,
        dropout=0.2,
        top_k=3,
        num_kernels=3,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.top_k = int(top_k)
        self.e_layers = int(e_layers)

        self.embed = nn.Linear(self.enc_in, int(d_model))
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    _InceptionBlock2D(int(d_model), int(d_ff), num_kernels),
                    nn.GELU(),
                    _InceptionBlock2D(int(d_ff), int(d_model), num_kernels),
                )
                for _ in range(self.e_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(int(d_model), self.c_out)

    def _fft_periods(self, x):
        # x: [B, T, d_model]
        xf = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(xf).mean(dim=0).mean(dim=-1)  # [freq]
        amplitude[0] = 0.0
        k = min(self.top_k, amplitude.shape[0] - 1)
        _, top_idx = torch.topk(amplitude, k)
        periods = (x.shape[1] // torch.clamp(top_idx, min=1)).detach().cpu().tolist()
        weights = torch.abs(xf).mean(dim=-1)[:, top_idx]  # [B, k]
        return [max(int(p), 1) for p in periods], weights

    def _times_block(self, x, block):
        B, T, D = x.shape
        periods, weights = self._fft_periods(x)
        outputs = []
        for period in periods:
            pad_len = (period - (T % period)) % period
            if pad_len:
                padded = torch.cat([x, x[:, -pad_len:, :]], dim=1)
            else:
                padded = x
            rows = padded.shape[1] // period
            grid = padded.reshape(B, rows, period, D).permute(0, 3, 1, 2)  # [B, D, rows, period]
            grid = block(grid)
            grid = grid.permute(0, 2, 3, 1).reshape(B, rows * period, D)
            outputs.append(grid[:, :T, :])
        stacked = torch.stack(outputs, dim=-1)  # [B, T, D, k]
        w = torch.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)  # [B,1,1,k]
        return (stacked * w).sum(dim=-1)

    def forward(self, x):
        x_enc = x.squeeze(1).permute(0, 2, 1)  # [B, L, C]
        means = x_enc.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        h = self.embed(x_norm)  # [B, L, d_model]
        h = self.predict_linear(h.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L+H, d_model]
        for block in self.blocks:
            h = self.layer_norm(h + self.dropout(self._times_block(h, block)))
        out = self.projection(h)  # [B, L+H, c_out]
        out = out[:, -self.pred_len:, :]
        out = out * stdev[:, :, : self.c_out] + means[:, :, : self.c_out]
        return out


class _SelectiveSSM(nn.Module):
    """Pure-PyTorch diagonal selective state-space scan (no mamba_ssm fused kernel).

    Input-dependent (selective) A/B/C/delta, sequential scan over the token axis.
    This approximates the Mamba S6 block in plain PyTorch so it runs without the
    fused CUDA kernel; the paper should label it as a PyTorch SSM approximation.
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.x_proj = nn.Linear(self.d_model, 2 * self.d_state + 1)
        self.dt_proj = nn.Linear(1, self.d_model)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x):  # x: [B, N, d_model]
        B, N, D = x.shape
        A = -torch.exp(self.A_log)  # [D, d_state]
        proj = self.x_proj(x)  # [B, N, 2*d_state+1]
        B_s, C_s, dt = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        delta = nn.functional.softplus(self.dt_proj(dt))  # [B, N, D]

        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(N):
            dt_t = delta[:, t, :].unsqueeze(-1)  # [B, D, 1]
            dA = torch.exp(dt_t * A.unsqueeze(0))  # [B, D, d_state]
            dB = dt_t * B_s[:, t, :].unsqueeze(1)  # [B, D, d_state]
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y = (h * C_s[:, t, :].unsqueeze(1)).sum(dim=-1)  # [B, D]
            ys.append(y)
        y = torch.stack(ys, dim=1)  # [B, N, D]
        return y + x * self.D


class SMambaBaseline(nn.Module):
    """Faithful lightweight S-Mamba-style baseline with the train.py IO contract.

    Keeps the defining S-Mamba design: variates are tokens, and a (bidirectional)
    selective state-space module mixes information across variate tokens, replacing
    the variate-attention of iTransformer. Uses a pure-PyTorch selective scan, so it
    runs without the mamba_ssm fused CUDA kernel (labeled as PyTorch SSM approximation).

    Input: [B, 1, C, L]. Output: [B, H, c_out].
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in=12,
        c_out=3,
        d_model=32,
        e_layers=2,
        d_state=16,
        dropout=0.2,
        bidirectional=True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.bidirectional = bool(bidirectional)

        self.value_embedding = nn.Linear(self.seq_len, int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.ssm_fwd = nn.ModuleList([_SelectiveSSM(int(d_model), int(d_state)) for _ in range(int(e_layers))])
        self.ssm_bwd = nn.ModuleList(
            [_SelectiveSSM(int(d_model), int(d_state)) for _ in range(int(e_layers))]
        ) if self.bidirectional else None
        self.norms = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(int(e_layers))])
        self.projection = nn.Linear(int(d_model), self.pred_len)

    def forward(self, x):
        x_enc = x.squeeze(1).permute(0, 2, 1)  # [B, L, C]
        means = x_enc.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        # variate tokens: [B, C, L] -> [B, C, d_model]
        tokens = self.value_embedding(x_norm.permute(0, 2, 1))
        h = self.dropout(tokens)
        for i, fwd in enumerate(self.ssm_fwd):
            out = fwd(h)
            if self.bidirectional:
                rev = self.ssm_bwd[i](torch.flip(h, dims=[1]))
                out = out + torch.flip(rev, dims=[1])
            h = self.norms[i](h + out)

        pred = self.projection(h)  # [B, C, pred_len]
        pred = pred.permute(0, 2, 1)  # [B, pred_len, C]
        pred = pred * stdev + means
        return pred[:, :, : self.c_out]
