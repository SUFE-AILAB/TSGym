import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, DSAttention
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding, PatchEmbedding
import numpy as np

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
class Model(nn.Module):
    def __init__(self, configs, norm_gym='None', input_embed_gym='series-patching', attn_gym='sparse-attention'):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.norm_gym = norm_gym
        self.input_embed_gym = input_embed_gym
        self.attn_gym = attn_gym
        # Normalization

        # Input Embedding
        if self.input_embed_gym == 'positional-encoding':
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

            self.decoder_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.input_embed_gym == 'series-patching':
            patch_len = 16
            stride = padding = 8
            # from PatchTST
            self.enc_embedding = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=configs.dropout)
            # todo: PatchTST is encoder-only
            self.dec_embedding = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride, padding=padding, dropout=configs.dropout)
            self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        # Attention Layer
        if self.attn_gym == 'self-attention':
            Attention = FullAttention
        elif self.attn_gym == 'auto-correlation':
            raise NotImplementedError
        elif self.attn_gym == 'sparse-attention':
            Attention = ProbAttention
        elif self.attn_gym == 'destationary-attention':
            Attention = DSAttention
        else:
            raise NotImplementedError
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attention(mask_flag=False, factor=configs.factor, attention_dropout=configs.dropout,
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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, verbose=False):
        # series normalization
        if self.norm_gym == 'None':
            pass
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        if verbose: print(f'The shape of x_enc: {x_enc.shape}')

        # encoder input embedding
        if self.input_embed_gym == 'positional-encoding':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
        elif self.input_embed_gym == 'series-patching':
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
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # decoder input embedding
        if self.input_embed_gym == 'positional-encoding':
            dec_out = self.dec_embedding(x_dec, x_mark_dec)
        elif self.input_embed_gym == 'series-patching':
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
        if self.input_embed_gym == 'positional-encoding':
            dec_out = self.decoder_projection(dec_out) # B x L x d_model -> B x L x D
            dec_out = dec_out[:, -self.pred_len:, :] # B x pred_len x D
        elif self.input_embed_gym == 'series-patching':
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
        if self.norm_gym == 'None':
            pass
        else:
            # De-Normalization from Non-stationary Transformer
            # B x D x pred_len
            dec_out = dec_out * \
                    (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
