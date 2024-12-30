import torch
import torch.nn as nn
import torch.nn.functional as F

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

# series decomposition
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# MoE series decomposition
# class series_decomp_multi(nn.Module):
#     """
#     Multiple Series decomposition block from FEDformer
#     """

#     def __init__(self, kernel_size):
#         super(series_decomp_multi, self).__init__()
#         self.kernel_size = kernel_size
#         self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

#     def forward(self, x):
#         moving_mean = []
#         res = []
#         for func in self.series_decomp:
#             sea, moving_avg = func(x)
#             moving_mean.append(moving_avg)
#             res.append(sea)

#         sea = sum(res) / len(res)
#         moving_mean = sum(moving_mean) / len(moving_mean)
#         return sea, moving_mean

# from: https://github.com/MAZiqing/FEDformer/blob/master/layers/Autoformer_EncDec.py
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 
    
# series decomposition using Discrete Fourier Transform (DFT)
# https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py#L9
class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf, n=x.shape[-1]) # modify
        x_trend = x - x_season
        return x_season, x_trend