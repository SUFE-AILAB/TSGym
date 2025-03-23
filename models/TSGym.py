import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, DSAttention, FourierCrossAttention, AutoCorrelation
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding, PatchEmbedding_wo_pos, PatchEmbedding
from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp
import numpy as np
from copy import deepcopy

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# todo: encoder-decoder architecture
# todo: non-stationary
class DNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(configs.d_model, configs.d_model),
                                nn.GELU(),
                                nn.Linear(configs.d_model, configs.d_model))

    def forward(self, x, attn_mask=None): # input shape: [BxSxD]
        x = self.fc(x)
        return x, None
    
class GRU(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder = nn.GRU(input_size=configs.d_model,
                              hidden_size=configs.d_model,
                              num_layers=configs.e_layers,
                              batch_first=True)

    def forward(self, x, attn_mask=None): # input shape: [BxSxD]
        x, _ = self.encoder(x)
        return x, None

class Model(nn.Module):
    def __init__(self, configs,
                  gym_series_norm=None,
                  gym_series_decomp=None,
                  gym_input_embed='series-patching',
                  gym_network_architecture='MLP',
                  gym_attn='sparse-attention',
                  gym_encoder_only=True,
                  ):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.gym_series_norm = gym_series_norm
        self.gym_series_decomp = eval(gym_series_decomp) if isinstance(gym_series_decomp, str) else gym_series_decomp
        self.gym_input_embed = gym_input_embed
        self.gym_network_architecture = gym_network_architecture
        self.gym_attn = gym_attn
        self.gym_encoder_only = eval(gym_encoder_only) if isinstance(gym_encoder_only, str) else gym_encoder_only

        # Series Normalization
        if self.gym_series_norm == 'None':
            self.series_norm = Normalize(configs.enc_in, affine=False, non_norm=True)
        elif self.gym_series_norm == 'Stat':
            self.series_norm = Normalize(configs.enc_in, affine=False, non_norm=False)
        elif self.gym_series_norm == 'RevIN':
            self.series_norm = Normalize(configs.enc_in, affine=True, non_norm=False)
        else:
            raise NotImplementedError

        # Series Decomposition
        print(f'moving avg for series decomposition: {configs.moving_avg}')
        self.series_decompsition = series_decomp(configs.moving_avg) if self.gym_series_decomp else None

        # Input Embedding
        if self.gym_input_embed == 'positional-encoding':
            if self.gym_network_architecture == 'Transformer':
                self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            else:
                self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

            self.decoder_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.head = nn.Linear(configs.seq_len, configs.pred_len)
        elif self.gym_input_embed == 'series-patching':
            patch_len = 16
            stride = padding = 8

            # patch_len = 3 if configs.seq_len % 3 == 0 else 4
            # stride = patch_len
            # padding = 0

            if self.gym_network_architecture == 'Transformer':
                # from PatchTST
                self.enc_embedding = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=configs.dropout)
                # todo: PatchTST is encoder-only
                self.dec_embedding = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=configs.dropout)
            else:
                self.enc_embedding = PatchEmbedding_wo_pos(d_model=configs.d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=configs.dropout)
                self.dec_embedding = PatchEmbedding_wo_pos(d_model=configs.d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=configs.dropout)

            # todo: 为什么原来是stride + 2
            self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
            # self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 1)
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        # Attention Layer
        if self.gym_attn == 'self-attention':
            Attention = FullAttention
        elif self.gym_attn == 'auto-correlation':
            Attention = AutoCorrelation
        elif self.gym_attn == 'sparse-attention':
            Attention = ProbAttention
        elif self.gym_attn == 'frequency-enhanced-attention':
            Attention = FourierCrossAttention
        elif self.gym_attn == 'destationary-attention':
            # https://github.com/thuml/Time-Series-Library/blob/3aed70eb3d7b8e5b51e8aafe1f9e69ed06d11de8/models/Nonstationary_Transformer.py#L113
            raise NotImplementedError
        else:
            if self.gym_network_architecture != 'Transformer':
                pass
            else:
                raise NotImplementedError
        
        # Encoder
        if self.gym_network_architecture == 'Transformer':
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attention(configs=configs,
                                                 mask_flag=False,
                                                 factor=configs.factor,
                                                 attention_dropout=configs.dropout, 
                                                 output_attention=False),
                                    configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            # Decoder
            if not self.gym_encoder_only:
                if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                    self.decoder = Decoder(
                        [
                            DecoderLayer(
                                AttentionLayer(
                                    Attention(mask_flag=True, factor=configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False), # masked multi-head attention
                                    configs.d_model, configs.n_heads),
                                AttentionLayer(
                                    Attention(mask_flag=False, factor=configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                                    configs.d_model, configs.n_heads),
                                configs.d_model,
                                configs.d_ff,
                                dropout=configs.dropout,
                                activation=configs.activation,
                            )
                            for l in range(configs.d_layers)
                        ],
                        norm_layer=torch.nn.LayerNorm(configs.d_model)
                    )
                else:
                    raise NotImplementedError
            else:
                self.decoder = None
        else:
            if not self.gym_encoder_only:
                raise NotImplementedError
            elif self.gym_network_architecture == 'MLP':
                self.encoder = DNN(configs)
            elif self.gym_network_architecture == 'GRU':
                self.encoder = GRU(configs)
            else:
                raise NotImplementedError

        # If series decomposition, deepcopy encoder to cosntruct seasonal & trend branches
        if self.series_decompsition:
            if not self.gym_encoder_only:
                raise NotImplementedError
            self.encoder_seasonal = deepcopy(self.encoder)
            self.encoder_trend = deepcopy(self.encoder)
            del self.encoder


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, verbose=False):
        # the input shape of x_enc: BxSxD
        if verbose: print(f'The shape of x_enc: {x_enc.shape}')

        # series normalization
        x_enc = self.series_norm(x_enc, 'norm')
        
        # series decomposition (todo整理代码, 目前还是太复杂了)
        if self.gym_series_decomp:
            if not self.gym_encoder_only:
                raise NotImplementedError
            # series decomposition
            seasonal_init, trend_init = self.series_decompsition(x_enc)
            # input encoding
            if self.gym_input_embed == 'positional-encoding':
                enc_out_seasonal = self.enc_embedding(seasonal_init, x_mark_enc)
                enc_out_trend = self.enc_embedding(trend_init, x_mark_enc)
            elif self.gym_input_embed == 'series-patching':
                enc_out_seasonal, n_vars = self.enc_embedding(seasonal_init.permute(0, 2, 1)) # BxSxD -> BxDxS
                enc_out_trend, n_vars = self.enc_embedding(trend_init.permute(0, 2, 1))
            else:
                raise NotImplementedError

            enc_out_seasonal, _ = self.encoder_seasonal(enc_out_seasonal, attn_mask=None)
            enc_out_trend, _ = self.encoder_trend(enc_out_trend, attn_mask=None)
            enc_out = enc_out_seasonal + enc_out_trend
            del enc_out_seasonal, enc_out_trend

            if self.gym_input_embed == 'positional-encoding':
                dec_out = self.decoder_projection(enc_out) # BxSxd_model -> BxSxD
                dec_out = self.head(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # BxSxD -> BxDxS -> BxDxpred_len -> Bxpred_lenxD
                dec_out = dec_out[:, -self.pred_len:, :] # B x pred_len x D
            elif self.gym_input_embed == 'series-patching':
                # z: [bs x nvars x patch_num x d_model]
                enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
                # z: [bs x nvars x d_model x patch_num]
                enc_out = enc_out.permute(0, 1, 3, 2)
                dec_out = self.head(enc_out)  # B x D x pred_len
                dec_out = dec_out.permute(0, 2, 1) # B x pred_len x D
            else:
                raise NotImplementedError
        else:
            # encoder input embedding
            if self.gym_input_embed == 'positional-encoding':
                enc_out = self.enc_embedding(x_enc, x_mark_enc)
            elif self.gym_input_embed == 'series-patching':
                # do patching and embedding
                x_enc = x_enc.permute(0, 2, 1) # BxSxD -> BxDxS
                # u: [bs * nvars x patch_num x d_model]
                enc_out, n_vars = self.enc_embedding(x_enc)
            else:
                raise NotImplementedError
            if verbose: print(f'The shape of enc_out after input encoding: {enc_out.shape}')

            # attention in encoder, the shape of enc_out
            # no patching: [bs x seq_len x d_model]
            # patching: [(bs * nvars) x patch_num x d_model]
            enc_out, _ = self.encoder(enc_out, attn_mask=None)

            if self.gym_encoder_only: # encoder-only
                if self.gym_input_embed == 'positional-encoding':
                    dec_out = self.decoder_projection(enc_out) # BxSxd_model -> BxSxD
                    # projection to predicting length
                    dec_out = self.head(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # BxSxD -> BxDxS -> BxDxpred_len -> Bxpred_lenxD
                    dec_out = dec_out[:, -self.pred_len:, :]
                elif self.gym_input_embed == 'series-patching':
                    # z: [bs x nvars x patch_num x d_model]
                    enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
                    # z: [bs x nvars x d_model x patch_num]
                    enc_out = enc_out.permute(0, 1, 3, 2)
                    dec_out = self.head(enc_out)  # B x D x pred_len
                    dec_out = dec_out.permute(0, 2, 1) # B x pred_len x D
                else:
                    raise NotImplementedError

            else: # encoder-decoder
                # decoder input embedding
                if self.gym_input_embed == 'positional-encoding':
                    dec_out = self.dec_embedding(x_dec, x_mark_dec)
                elif self.gym_input_embed == 'series-patching':
                    # do patching and embedding
                    x_dec = x_dec.permute(0, 2, 1) # BxSxD -> BxDxS
                    # u: [(bs * nvars) x patch_num x d_model]
                    dec_out, n_vars = self.dec_embedding(x_dec)
                else:
                    raise NotImplementedError
                if verbose: print(f'The shape of x_dec: {x_dec.shape}')
                
                # attention in decoder
                dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
                if verbose: print(f'The shape of dec_out: {dec_out.shape}')

                # output layer
                if self.gym_input_embed == 'positional-encoding':
                    dec_out = self.decoder_projection(dec_out) # B x L x d_model -> B x L x D
                    dec_out = dec_out[:, -self.pred_len:, :] # B x pred_len x D
                elif self.gym_input_embed == 'series-patching':
                    # z: [bs x nvars x patch_num x d_model]
                    dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
                    # z: [bs x nvars x d_model x patch_num]
                    dec_out = dec_out.permute(0, 1, 3, 2)

                    # Decoder
                    dec_out = self.head(dec_out)  # B x D x pred_len
                    dec_out = dec_out.permute(0, 2, 1) # B x pred_len x D
                else:
                    raise NotImplementedError

        # de-normalization layer (if necessary)
        dec_out = self.series_norm(dec_out, 'denorm')        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
