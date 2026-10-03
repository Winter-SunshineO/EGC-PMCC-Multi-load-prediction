import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """Minimal DLinear point baseline.

    Input: [B, 1, C, L]. Output: [B, H, out_nodes].
    """

    def __init__(self, seq_len, pred_len, enc_in=12, c_out=3, individual=False, kernel_size=25):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.individual = bool(individual)
        self.decomp = SeriesDecomp(kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.c_out)])
            self.linear_trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.c_out)])
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        x = x[:, :, : self.c_out]
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            outputs = []
            for idx in range(self.c_out):
                y = self.linear_seasonal[idx](seasonal[:, idx, :])
                y = y + self.linear_trend[idx](trend[:, idx, :])
                outputs.append(y)
            y = torch.stack(outputs, dim=-1)
        else:
            y = self.linear_seasonal(seasonal) + self.linear_trend(trend)
            y = y.permute(0, 2, 1)
        return y


class NLinear(nn.Module):
    """Minimal NLinear point baseline.

    Input: [B, 1, C, L]. Output: [B, H, out_nodes].
    """

    def __init__(self, seq_len, pred_len, enc_in=12, c_out=3, individual=False):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.c_out = int(c_out)
        self.individual = bool(individual)

        if self.individual:
            self.linear = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.c_out)])
        else:
            self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        x = x[:, :, : self.c_out]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = x.permute(0, 2, 1)

        if self.individual:
            outputs = [self.linear[idx](x[:, idx, :]) for idx in range(self.c_out)]
            y = torch.stack(outputs, dim=-1)
        else:
            y = self.linear(x).permute(0, 2, 1)
        return y + seq_last[:, :, : self.c_out]
