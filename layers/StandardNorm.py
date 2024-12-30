import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

# source code from: https://github.com/weifantt/Dish-TS/tree/master
class DishTS(nn.Module):
    def __init__(self, configs, init='standard', mode='DishTS'):
        '''
        :param configs:
        :param init: 'standard', 'avg' or 'uniform'
        '''
        super().__init__()
        activate = True
        self.mode = mode
        n_series = configs.enc_in # number of series
        lookback = configs.seq_len  # lookback length

        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2) / lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2) / lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(n_series, lookback, 2) / lookback + torch.rand(n_series, lookback, 2) / lookback)

        # todo: 这里加了一个linear projection, 因为RevIN和Dish-TS都是假设输入序列和输出序列BxSxD的D维是相同的, 但是实际使用过程中往往是不同的, 例如输入是multivariate, 输出是univariate
        # todo: 还有一个问题是在何时反标准化, RevIN和Dish-TS是对输出序列反标准化, 但现在的输出是univairate, 并且one-step forecasting. 所以暂时在flatten之前反标准化
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

        # 同样地, Dish-TS中的xih和phih也需要projection, xih需要保证为正, 额外用了softplus

    def forward(self, batch_x, mode=None, dec_inp=None):
        if mode == 'norm':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            if dec_inp is None:
                return batch_x
            else:
                return batch_x, dec_inp
        elif mode == 'denorm':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y
        else:
            raise NotImplementedError

    def preget(self, batch_x):
        if self.mode == 'RevIN':
            self.phil = self.phih = torch.mean(batch_x, axis=1, keepdim=True).detach()
            self.xil = self.xih = torch.var(batch_x, axis=1, keepdim=True).detach()

        elif self.mode == 'DishTS':
            x_transpose = batch_x.permute(2, 0, 1)
            theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
            if self.activate:
                theta = F.gelu(theta)
            self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
            # 注意lookback window和forecast horizon的xil/xih都是根据phil/phih自动算出来的, 而不是模型学习额外再学习的
            self.xil = torch.sum(torch.pow(batch_x - self.phil, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)
            self.xih = torch.sum(torch.pow(batch_x - self.phih, 2), axis=1, keepdim=True) / (batch_x.shape[1] - 1)
        else:
            raise NotImplementedError

    def forward_process(self, batch_input):
        # print(batch_input.shape, self.phil.shape, self.xih.shape)
        # 参数phil和xil是paper中proposed需要学习的参数, 详见paper公式(6)
        temp = (batch_input - self.phil) / torch.sqrt(self.xil + 1e-8)
        # 这里参数gamma与beta实际上from RevIN, 详见paper中的公式(2), https://openreview.net/pdf?id=cGDAkQo1C0p
        # 可以理解为对标准化到高斯分布的变量做峰度和偏度的调整, 只不过Dish-TS做标准化的统计量from NN output
        # 确认一下self.gamma和self.beta维数是否正确, 现在只有一维
        rst = temp.mul(self.gamma) + self.beta
        return rst

    def inverse_process(self, batch_input):
        # 为forward process逆操作
        return (((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih)

# todo: add RevIN normalization module, see
# https://github.com/weifantt/Dish-TS/blob/e674d3b94b832491f63a533d60e40a75031d2c75/REVIN.py#L9
# todo: add codes for encoder-decoder Transformer
# todo: 为什么Dish-TS的code里面并没有在loss后面加上对于均值的regularization(详见Paper公式9)
class DishTs_Wrapped_Model(nn.Module):
    def __init__(self, forecast_model, norm_model):
        super().__init__()
        self.fm = forecast_model
        self.nm = norm_model

    def forward(self, x, x_dec=None):
        # forward normalization
        x = self.nm(x, mode='norm')
        # forecasting
        x = self.fm(x)
        # inverse normalization
        pred = self.nm(x, mode='denorm')

        return pred