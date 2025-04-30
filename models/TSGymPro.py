import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, DSAttention, FourierCrossAttention, AutoCorrelation
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding, DataEmbedding_inverted, PatchEmbedding_wo_pos, PatchEmbedding
from layers.StandardNorm import Normalize, DishTS
from layers.SeriesDecom import series_decomp, series_decomp_multi, DFT_series_decomp
import numpy as np
from copy import deepcopy
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from models import TimeLLM, Moment


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

def calculate_patch_num(seq_len, patch_len=8):
    # 原本stride=1, 例如seq_len=60会产生60-patch_len+1个patches, 高度overlapped
    # 现在stride=patch_len, 产生的是non-overlapped patches
    stride = patch_len
    padding = patch_len - seq_len % patch_len if seq_len % patch_len != 0 else 0
    patch_num = int((seq_len + padding - patch_len) / stride + 1)
    return stride, padding, patch_num, patch_len

# todo: encoder-decoder architecture
# todo: non-stationary
class DNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(configs.d_model, configs.d_model),
                                nn.GELU(),
                                nn.Linear(configs.d_model, configs.d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        x = self.fc(x)
        return x, None
    
class GRU(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder = nn.GRU(input_size=configs.d_model,
                              hidden_size=configs.d_model,
                              num_layers=configs.e_layers,
                              batch_first=True)

    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        x, _ = self.encoder(x)
        return x, None
    
class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y

# GPT4TS, TimeLLM
class LLM(nn.Module):
    def __init__(self, configs,
                 network_architecture,
                 frozen=True, gpt_layers=6):
        super().__init__()
        self.network_architecture = network_architecture

        if network_architecture == 'LLM-GPT4TS':
            self.encoder = GPT2Model.from_pretrained('./models/llm/gpt2', output_attentions=False, output_hidden_states=True)
            self.encoder.h = self.encoder.h[:gpt_layers]
            if frozen:
                for i, (name, param) in enumerate(self.encoder.named_parameters()):
                    if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                        param.requires_grad = True
                    elif 'mlp' in name and configs.mlp == 1:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                pass
            self.proj = nn.Linear(768, configs.d_model) # todo: 截断还是linear projection?
        elif network_architecture == 'LLM-TimeLLM':
            self.encoder = TimeLLM.Model(configs, frozen=frozen)
        else:
            raise NotImplementedError
    
    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        if self.network_architecture == 'LLM-GPT4TS':
            # padding zero on the last dimension, BxTx768
            x = torch.nn.functional.pad(x, (0, 768 - x.shape[-1]))
            x = self.encoder(inputs_embeds=x).last_hidden_state
            x = self.proj(x)
        elif self.network_architecture == 'LLM-TimeLLM':
            x = self.encoder(x)
        else:
            raise NotImplementedError
        return x, None

# Timer, Moment
class TSFM(nn.Module):
    def __init__(self, configs,
                 network_architecture,
                 frozen=True):
        super().__init__()
        self.network_architecture = network_architecture

        if network_architecture == 'TSFM-Timer':
            # Timer里面的decoder实际上是encoder
            self.encoder = Encoder(attn_layers=[EncoderLayer(AttentionLayer(FullAttention(configs=configs,
                                                                                        mask_flag=False,
                                                                                        factor=configs.factor,
                                                                                        attention_dropout=configs.dropout, 
                                                                                        output_attention=False),
                                                                            configs.d_model, configs.n_heads),
                                                            configs.d_model,
                                                            configs.d_ff,
                                                            dropout=configs.dropout,
                                                            activation=configs.activation) for _ in range(configs.e_layers)],
                                norm_layer=torch.nn.LayerNorm(configs.d_model))
            sd = torch.load('./models/llm/timer/Timer_forecast_1.0.ckpt', map_location="cpu", weights_only=False)["state_dict"]
            for k in sd.keys():
                print(k)
            sd = {k[6:]: v for k, v in sd.items() if 'decoder' in k}
            self.encoder.load_state_dict(sd, strict=False)
            if frozen:
                # frozen attention weights
                for name, param in self.encoder.named_parameters():
                    # 只finetune attention layer中的layer norm仿射变换的参数
                    if 'attn_layers' in name and 'norm' not in name:
                        param.requires_grad = False
            else:
                pass
        elif network_architecture == 'TSFM-Moment':
            self.encoder = Moment.Model(frozen=frozen)
            self.proj = nn.Linear(768, configs.d_model)
        else:
            raise NotImplementedError

    def forward(self, x, attn_mask=None, tau=None, delta=None): # input shape: [BxSxD]
        if self.network_architecture == 'TSFM-Moment':
            x = torch.nn.functional.pad(x, (0, 768 - x.shape[-1])) # padding to 768
            x, _ = self.encoder(x)
            x = self.proj(x) # 768 -> d_model
        elif self.network_architecture == 'TSFM-Timer':
            x, _ = self.encoder(x)
        else:
            raise NotImplementedError
        return x, None


class Model(nn.Module):
    def __init__(self, configs,
                  gym_series_sampling=False,
                  gym_series_norm=None,
                  gym_series_decomp=None,
                  gym_channel_independent=False,
                  gym_input_embed='series-encoding',
                  gym_network_architecture='Transformer',
                  gym_attn='self-attention',
                  gym_feature_attn='self-attention',
                  gym_encoder_only=True,
                  gym_frozen=True,
                  ):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.configs = configs

        self.gym_series_sampling = eval(gym_series_sampling) if isinstance(gym_series_sampling, str) else gym_series_sampling
        self.gym_series_norm = gym_series_norm
        self.gym_series_decomp = gym_series_decomp
        self.gym_channel_independent = eval(gym_channel_independent) if isinstance(gym_channel_independent, str) else gym_channel_independent
        self.gym_input_embed = gym_input_embed
        self.gym_network_architecture = gym_network_architecture
        self.gym_attn = gym_attn
        self.gym_feature_attn = gym_feature_attn
        self.gym_encoder_only = eval(gym_encoder_only) if isinstance(gym_encoder_only, str) else gym_encoder_only
        self.gym_frozen = eval(gym_frozen) if isinstance(gym_frozen, str) else gym_frozen
        # pipeline
        # ↓ series sampling
        # ↓ series normalization
            # None
            # Stat
            # RevIN
            # DishTS
        # ↓ series decomposition
            # None
            # Moving Avarage
            # MoE Moving Average
            # DFT
        # ↓ series embedding (tokenization)
            # channel-dependent
                # series-encoding
            # channel-independent
                # series-encoding
                # series-patching
        # ↓ series encoding
            # MLP
            # RNN
            # Transformer
                # Vanilla Transformer
                # Informer
                # Autoformer
                # FEDformer
                # ...
        # ↓ series denormalization
        #   output

        # Series Sampling (multi-granularity)
        if self.gym_series_sampling:
            self.series_sampling = self.multi_scale_process_inputs
        else:
            self.series_sampling = None

        # Series Normalization
        if self.gym_series_norm == 'None':
            self.series_norm = Normalize(configs.enc_in, affine=False, non_norm=True)
        elif self.gym_series_norm == 'Stat':
            self.series_norm = Normalize(configs.enc_in, affine=False, non_norm=False)
        elif self.gym_series_norm == 'RevIN':
            self.series_norm = Normalize(configs.enc_in, affine=True, non_norm=False)
        elif self.gym_series_norm == 'DishTS':
            self.series_norm = DishTS(configs)
        else:
            raise NotImplementedError
        
        if self.series_sampling: # series normalization有可学习参数
            if self.gym_series_norm == 'DishTS':
                self.series_norm = nn.ModuleList([DishTS(configs, seq_len=configs.seq_len // (configs.down_sampling_window ** i)) 
                                                  for i in range(configs.down_sampling_layers + 1)])
            else:
                self.series_norm = nn.ModuleList(deepcopy(self.series_norm) 
                                                 for i in range(self.configs.down_sampling_layers + 1))

        # Series Decomposition
        print(f'series decomposition: {configs.moving_avg}')
        if self.gym_series_decomp == 'None':
            self.series_decompsition = None
        elif self.gym_series_decomp == 'MA':
            self.series_decompsition = series_decomp(configs.moving_avg)
        elif self.gym_series_decomp == 'MoEMA':
            self.series_decompsition = series_decomp_multi([15, 25, 35])
        elif self.gym_series_decomp == 'DFT':
            self.series_decompsition = DFT_series_decomp()
        else:
            raise NotImplementedError
        
        # Series Tokenization
        # no-patching: BxSxD -> (BxD)xSx1; patching: BxSxD -> (BxD)xS -> (BxD) x patch_num x patch_len
        # https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py
        if self.gym_channel_independent: # channel-independent
            if self.gym_input_embed == 'series-encoding':
                if self.gym_network_architecture in ['MLP', 'GRU']:
                    self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
                    self.dec_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
                else:
                    self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
                    self.dec_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)

                if self.series_sampling:
                    self.enc_embedding = nn.ModuleList(deepcopy(self.enc_embedding) for i in range(self.configs.down_sampling_layers + 1))
                    self.dec_embedding = nn.ModuleList(deepcopy(self.dec_embedding) for i in range(self.configs.down_sampling_layers + 1))
                    self.decoder_projection = nn.ModuleList(nn.Linear(configs.d_model, 1, bias=True)
                                              for i in range(self.configs.down_sampling_layers + 1))
                    self.head = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ])
                else:
                    self.decoder_projection = nn.Linear(configs.d_model, 1, bias=True)
                    self.head = nn.Linear(configs.seq_len, configs.pred_len)
            elif self.gym_input_embed == 'series-patching':
                stride, padding, patch_num, patch_len = calculate_patch_num(configs.seq_len)

                if self.gym_network_architecture in ['MLP', 'GRU']:
                    self.enc_embedding = PatchEmbedding_wo_pos(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                               padding=padding, dropout=configs.dropout)
                    self.dec_embedding = PatchEmbedding_wo_pos(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                               padding=padding, dropout=configs.dropout)
                else:
                    # from PatchTST (todo: PatchTST is encoder-only)
                    self.enc_embedding = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                        padding=padding, dropout=configs.dropout)
                    self.dec_embedding = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                        padding=padding, dropout=configs.dropout)
                    
                # self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
                self.head = FlattenHead(configs.enc_in, configs.d_model * patch_num, configs.pred_len, head_dropout=configs.dropout)

                if self.series_sampling:
                    self.enc_embedding, self.dec_embedding, self.head = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
                    for i in range(configs.down_sampling_layers + 1):
                        seq_len_ = configs.seq_len // (configs.down_sampling_window ** i)
                        stride, padding, patch_num, patch_len = calculate_patch_num(seq_len_)
                        if self.gym_network_architecture in ['MLP', 'GRU']:
                            enc_embedding_ = PatchEmbedding_wo_pos(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                                   padding=padding, dropout=configs.dropout)
                            dec_embedding_ = PatchEmbedding_wo_pos(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                                   padding=padding, dropout=configs.dropout)
                        else:
                            enc_embedding_ = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                            padding=padding, dropout=configs.dropout)
                            dec_embedding_ = PatchEmbedding(d_model=configs.d_model, patch_len=patch_len, stride=stride,
                                                            padding=padding, dropout=configs.dropout)
                            
                        self.enc_embedding.append(enc_embedding_)
                        self.dec_embedding.append(dec_embedding_)
                        self.head.append(FlattenHead(configs.enc_in, configs.d_model * patch_num, configs.pred_len, head_dropout=configs.dropout))
                    
                    self.enc_embedding = nn.ModuleList(self.enc_embedding)
                    self.dec_embedding = nn.ModuleList(self.dec_embedding)
                    self.head = nn.ModuleList(self.head)
            else:
                raise NotImplementedError
        else: # channel-dependent
            if self.gym_input_embed == 'inverted-encoding':
                # inverted DataEmbedding w/o positional encoding
                self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
                self.dec_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
                # predicting head
                self.decoder_projection = None
                self.head = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            elif self.gym_input_embed == 'series-encoding':
                if self.gym_network_architecture in ['MLP', 'GRU']:
                    self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                    self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                else:
                    self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                    self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                    
                # decoder projection
                self.decoder_projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
                # predicting head
                self.head = nn.Linear(configs.seq_len, configs.pred_len)
            else:
                raise NotImplementedError

            if self.series_sampling:
                if self.gym_input_embed == 'inverted-encoding': raise NotImplementedError
                self.enc_embedding = nn.ModuleList(deepcopy(self.enc_embedding) for i in range(self.configs.down_sampling_layers + 1))
                self.dec_embedding = nn.ModuleList(deepcopy(self.dec_embedding) for i in range(self.configs.down_sampling_layers + 1))
                self.decoder_projection = nn.ModuleList(deepcopy(self.decoder_projection) for i in range(self.configs.down_sampling_layers + 1))
                self.head = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

        # multi-granularity interation (based-on self-attention)
        # seasonal: high -> low; trend: low -> high
        if self.series_sampling and self.series_decompsition:
            self.seasonal_mixing = AttentionLayer(
                                    FullAttention(mask_flag=False,
                                                  factor=configs.factor,
                                                  attention_dropout=configs.dropout,
                                                  output_attention=False), # masked multi-head attention
                                    configs.d_model, configs.n_heads)
            self.trend_mixing = AttentionLayer(
                                    FullAttention(mask_flag=False,
                                                  factor=configs.factor,
                                                  attention_dropout=configs.dropout,
                                                  output_attention=False), # masked multi-head attention
                                    configs.d_model, configs.n_heads)
            
            self.seasonal_mixing = nn.ModuleList([deepcopy(self.seasonal_mixing) 
                                                  for i in range(self.configs.down_sampling_layers)])
            self.trend_mixing = nn.ModuleList([deepcopy(self.trend_mixing) 
                                               for i in range(self.configs.down_sampling_layers)])
    
        # feature attention
        if self.gym_feature_attn == 'null':
            self.feature_encoder = None
        else:
            if self.gym_feature_attn == 'self-attention':
                FeatureAttention = FullAttention
            elif self.gym_feature_attn == 'sparse-attention':
                FeatureAttention = ProbAttention
            elif self.gym_feature_attn == 'frequency-enhanced-attention':
                FeatureAttention = FourierCrossAttention
            else:
                raise NotImplementedError

            # Encoder
            self.feature_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.feature_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(FeatureAttention(configs=configs,
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
            if self.series_sampling:
                self.feature_embedding = nn.ModuleList(deepcopy(self.feature_embedding) for i in range(self.configs.down_sampling_layers + 1))
                self.feature_encoder = nn.ModuleList(deepcopy(self.feature_encoder) for i in range(self.configs.down_sampling_layers + 1))


        # encoder(-decoder)                        
        if self.gym_network_architecture == 'Transformer':
            # Attention
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
                if self.gym_series_norm != 'Stat': raise NotImplementedError
                if self.gym_input_embed != 'series-encoding': raise NotImplementedError
                Attention = DSAttention

                if self.gym_series_sampling:
                    self.tau_learner = nn.ModuleList([Projector(enc_in=1 if self.gym_channel_independent else configs.enc_in, seq_len=configs.seq_len // (configs.down_sampling_window ** i),
                                                                hidden_dims=configs.p_hidden_dims,
                                                                hidden_layers=configs.p_hidden_layers,
                                                                output_dim=1)
                                                                for i in range(configs.down_sampling_layers + 1)])
                    self.delta_learner = nn.ModuleList([Projector(enc_in=1 if self.gym_channel_independent else configs.enc_in, seq_len=configs.seq_len // (configs.down_sampling_window ** i),
                                                                  hidden_dims=configs.p_hidden_dims,
                                                                  hidden_layers=configs.p_hidden_layers,
                                                                  output_dim=configs.seq_len // (configs.down_sampling_window ** i)) 
                                                                  for i in range(configs.down_sampling_layers + 1)])             
                
                else:
                    self.tau_learner = Projector(enc_in=1 if self.gym_channel_independent else configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                                hidden_layers=configs.p_hidden_layers, output_dim=1)
                    self.delta_learner = Projector(enc_in=1 if self.gym_channel_independent else configs.enc_in, seq_len=configs.seq_len,
                                                hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                                output_dim=configs.seq_len)
            else:
                raise NotImplementedError
            
            # Encoder
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
        elif 'LLM' in self.gym_network_architecture:
            # loads a pretrained GPT-2 base model
            self.encoder = LLM(configs, network_architecture=self.gym_network_architecture, frozen=self.gym_frozen)
            self.decoder = None
        elif 'TSFM' in self.gym_network_architecture:
            self.encoder = TSFM(configs, network_architecture=self.gym_network_architecture, frozen=self.gym_frozen)
            self.decoder = None
        else:
            if not self.gym_encoder_only:
                raise NotImplementedError
            elif self.gym_network_architecture == 'MLP':
                self.encoder = DNN(configs)
                self.decoder = None
            elif self.gym_network_architecture == 'GRU':
                self.encoder = GRU(configs)
                self.decoder = None
            else:
                raise NotImplementedError

        # If series decomposition, deepcopy encoder to cosntruct seasonal & trend branches
        if self.series_decompsition:
            if not self.gym_encoder_only:
                raise NotImplementedError
            self.encoder_seasonal = deepcopy(self.encoder)
            self.encoder_trend = deepcopy(self.encoder)
            if self.series_sampling:
                self.encoder_seasonal = nn.ModuleList(deepcopy(self.encoder_seasonal) for i in range(self.configs.down_sampling_layers + 1))
                self.encoder_trend = nn.ModuleList(deepcopy(self.encoder_trend) for i in range(self.configs.down_sampling_layers + 1))
            del self.encoder
        else:
            if self.series_sampling:
                self.encoder = nn.ModuleList(deepcopy(self.encoder) for i in range(self.configs.down_sampling_layers + 1))
                if self.decoder is not None:
                    self.decoder = nn.ModuleList(deepcopy(self.decoder) for i in range(self.configs.down_sampling_layers + 1))
            else:
                pass

    def multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, verbose=False):
        # the input shape of x_enc: BxSxD
        if verbose: print(f'The shape of x_enc: {x_enc.shape}')
        B, S, D = x_enc.shape

        # series sampling
        if self.series_sampling: x_enc, x_mark_enc = self.series_sampling(x_enc, x_mark_enc)

        # series normalization
        if isinstance(x_enc, list):
            x_enc = [self.series_norm[i](_, 'norm') for i, _ in enumerate(x_enc)]
        else:
            x_enc = self.series_norm(x_enc, 'norm')

        # feature attention
        if self.gym_feature_attn != 'null':
            if isinstance(x_enc, list):
                x_enc_fa = [self.feature_embedding[i](_, x_mark_enc[i]) for i, _ in enumerate(x_enc)]
                enc_out_fa = [self.feature_encoder[i](_)[0] for i, _ in enumerate(x_enc_fa)]
            else:
                x_enc_fa = self.feature_embedding(x_enc, x_mark_enc)
                enc_out_fa, _ = self.feature_encoder(x_enc_fa)
            

        if self.gym_channel_independent and self.gym_input_embed  == 'series-encoding':
            # BxSxD -> (BxD)xSx1
            if isinstance(x_enc, list):
                x_enc = [_.permute(0, 2, 1).contiguous().reshape(B * D, _.shape[1], 1) for _ in x_enc]
                if x_mark_enc is not None: x_mark_enc = [_.repeat(D, 1, 1) for _ in x_mark_enc]
            else:
                x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * D, S, 1)
                if x_mark_enc is not None: x_mark_enc = x_mark_enc.repeat(D, 1, 1)

        # learn tau and delta for non-stationary attention (if necessary)
        if isinstance(x_enc, list):
            if self.gym_attn == 'destationary-attention':
                tau, delta = [], []
                for i, x_enc_ in enumerate(x_enc):
                    mean_enc = x_enc_.mean(1, keepdim=True).detach()
                    std_enc = torch.sqrt(torch.var(x_enc_, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                    tau.append(self.tau_learner[i](x_enc_.clone().detach(), std_enc).exp())
                    delta.append(self.delta_learner[i](x_enc_.clone().detach(), mean_enc))
            else:
                tau, delta = [None] * len(x_enc), [None] * len(x_enc)
        else:
            if self.gym_attn == 'destationary-attention':
                mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
                std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
                # B x S x E, B x 1 x E -> B x 1, positive scalar
                tau = self.tau_learner(x_enc.clone().detach(), std_enc).exp() # x_raw
                # B x S x E, B x 1 x E -> B x S
                delta = self.delta_learner(x_enc.clone().detach(), mean_enc)
                if verbose: print(f'the shape of tau: {tau.shape}, delta: {delta.shape}')
            else:
                tau, delta = None, None
        
        # series decomposition (todo整理代码, 目前还是太复杂了)
        if self.series_decompsition:
            if not self.gym_encoder_only:
                raise NotImplementedError
            
            if self.series_sampling:
                # series decomposition
                seasonal_init, trend_init = [], []
                for _ in x_enc:
                    seasonal_init_, trend_init_ = self.series_decompsition(_)
                    seasonal_init.append(seasonal_init_); trend_init.append(trend_init_)
                # input encoding
                enc_out_seasonal, enc_out_trend = [], []
                for i, (seasonal_init_, trend_init_) in enumerate(zip(seasonal_init, trend_init)):
                    if self.gym_input_embed == 'series-encoding':
                        enc_out_seasonal_ = self.enc_embedding[i](seasonal_init_, x_mark_enc[i]) # todo: x_mark
                        enc_out_trend_ = self.enc_embedding[i](trend_init_, x_mark_enc[i])
                    elif self.gym_input_embed == 'series-patching':
                        enc_out_seasonal_, n_vars = self.enc_embedding[i](seasonal_init_.permute(0, 2, 1)) # BxSxD -> BxDxS
                        enc_out_trend_, n_vars = self.enc_embedding[i](trend_init_.permute(0, 2, 1))
                    else:
                        raise NotImplementedError
                    enc_out_seasonal.append(enc_out_seasonal_); enc_out_trend.append(enc_out_trend_)

                # multi-granularity mixing
                # seasonal: high -> low
                enc_out_seasonal_mixed = []
                for i, enc_out_seasonal_ in enumerate(enc_out_seasonal):
                    if i == 0:
                        enc_out_seasonal_mixed.append(enc_out_seasonal_)
                    else:
                        enc_out_seasonal_ = self.seasonal_mixing[i-1](queries=enc_out_seasonal_,
                                                                      keys=enc_out_seasonal_previous_,
                                                                      values=enc_out_seasonal_previous_,
                                                                      attn_mask=None)[0]
                        enc_out_seasonal_mixed.append(enc_out_seasonal_)
                    enc_out_seasonal_previous_ = enc_out_seasonal_
                enc_out_seasonal = enc_out_seasonal_mixed; del enc_out_seasonal_mixed
                
                # tred: low -> high
                enc_out_trend_mixed = []
                for i, enc_out_trend_ in enumerate(enc_out_trend[::-1]):
                    if i == 0:
                        enc_out_trend_mixed.append(enc_out_trend_)
                    else:
                        enc_out_trend_ = self.trend_mixing[i-1](queries=enc_out_trend_,
                                                                keys=enc_out_trend_previous_,
                                                                values=enc_out_trend_previous_,
                                                                attn_mask=None)[0]
                        enc_out_trend_mixed.append(enc_out_trend_)
                    enc_out_trend_previous_ = enc_out_trend_
                enc_out_trend = enc_out_trend_mixed[::-1]; del enc_out_trend_mixed

                # seasonal + trend
                enc_out_seasonal = [self.encoder_seasonal[i](_, attn_mask=None, tau=tau[i], delta=delta[i])[0] for i, _ in enumerate(enc_out_seasonal)]
                enc_out_trend = [self.encoder_trend[i](_, attn_mask=None, tau=tau[i], delta=delta[i])[0] for i, _ in enumerate(enc_out_trend)]
                enc_out = [enc_out_seasonal_ + enc_out_trend_ for enc_out_seasonal_, enc_out_trend_
                            in zip(enc_out_seasonal, enc_out_trend)]
                
                if self.gym_feature_attn != 'null': 
                    enc_out = [enc_out_ + enc_out_fa_ for enc_out_, enc_out_fa_ in zip(enc_out, enc_out_fa)]
                del enc_out_seasonal, enc_out_trend
            else:
                # series decomposition
                seasonal_init, trend_init = self.series_decompsition(x_enc)
                # input encoding
                if self.gym_input_embed == 'inverted-encoding':
                    enc_out_seasonal = self.enc_embedding(seasonal_init, x_mark_enc)
                    enc_out_trend = self.enc_embedding(trend_init, x_mark_enc)
                elif self.gym_input_embed == 'series-encoding':
                    enc_out_seasonal = self.enc_embedding(seasonal_init, x_mark_enc)
                    enc_out_trend = self.enc_embedding(trend_init, x_mark_enc)
                elif self.gym_input_embed == 'series-patching':
                    enc_out_seasonal, n_vars = self.enc_embedding(seasonal_init.permute(0, 2, 1)) # BxSxD -> BxDxS
                    enc_out_trend, n_vars = self.enc_embedding(trend_init.permute(0, 2, 1))
                else:
                    raise NotImplementedError

                # seasonal + trend
                enc_out_seasonal, _ = self.encoder_seasonal(enc_out_seasonal, attn_mask=None, tau=tau, delta=delta)
                enc_out_trend, _ = self.encoder_trend(enc_out_trend, attn_mask=None, tau=tau, delta=delta)
                enc_out = enc_out_seasonal + enc_out_trend
                if self.gym_feature_attn != 'null': enc_out = enc_out + enc_out_fa
                del enc_out_seasonal, enc_out_trend

            if self.gym_input_embed == 'inverted-encoding':
                dec_out = self.head(enc_out) # BxDxd_model -> BxDxpred_len
                dec_out = dec_out.permute(0, 2, 1) # BxDxpred_len -> Bxpred_lenxD
                # truncate if have covariate input, see: 
                # https://github.com/thuml/Time-Series-Library/blob/cdf8f0c3c5e79c1e8152e71dc35009ae46a6a920/layers/Embed.py#L141
                dec_out = dec_out[:, :, :D]
            elif self.gym_input_embed == 'series-encoding':
                if self.series_sampling:
                    dec_out = [self.decoder_projection[i](_) for i, _ in enumerate(enc_out)]
                    dec_out = [self.head[i](_.permute(0, 2, 1)).permute(0, 2, 1) for i, _ in enumerate(dec_out)]
                    dec_out = [_[:, -self.pred_len:, :] for _ in dec_out]
                    dec_out = torch.stack(dec_out, dim=-1).sum(-1)
                else:
                    dec_out = self.decoder_projection(enc_out) # BxSxd_model -> BxSxD
                    dec_out = self.head(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # BxSxD -> BxDxS -> BxDxpred_len -> Bxpred_lenxD
                    dec_out = dec_out[:, -self.pred_len:, :] # B x pred_len x D
                if self.gym_channel_independent: dec_out = dec_out.reshape(B, D, self.pred_len).permute(0, 2, 1).contiguous()
            elif self.gym_input_embed == 'series-patching':
                if self.series_sampling:
                    enc_out = [torch.reshape(_, (-1, n_vars, _.shape[-2], _.shape[-1])) for _ in enc_out]
                    enc_out = [_.permute(0, 1, 3, 2) for _ in enc_out]
                    dec_out = [self.head[i](_) for i, _ in enumerate(enc_out)]
                    dec_out = [_.permute(0, 2, 1) for _ in dec_out]
                    dec_out = torch.stack(dec_out, dim=-1).sum(-1)
                else:
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
            if self.gym_input_embed == 'inverted-encoding':
                enc_out = self.enc_embedding(x_enc, x_mark_enc)
            elif self.gym_input_embed == 'series-encoding':
                if self.series_sampling:
                    assert len(x_enc) == len(x_mark_enc)
                    enc_out = [self.enc_embedding[i](x_enc[i], x_mark_enc[i]) for i in range(len(x_enc))]
                else:
                    enc_out = self.enc_embedding(x_enc, x_mark_enc)
            elif self.gym_input_embed == 'series-patching':
                if self.series_sampling:
                    x_enc = [_.permute(0, 2, 1) for _ in x_enc]
                    enc_out = []
                    for i, x_enc_ in enumerate(x_enc):
                        enc_out_, n_vars = self.enc_embedding[i](x_enc_)
                        enc_out.append(enc_out_)
                else:
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
            if isinstance(enc_out, list):
                enc_out = [self.encoder[i](_, attn_mask=None, tau=tau[i], delta=delta[i])[0] for i, _ in enumerate(enc_out)]
                if self.gym_feature_attn != 'null': 
                    enc_out = [enc_out_ + enc_out_fa_ for enc_out_, enc_out_fa_ in zip(enc_out, enc_out_fa)]
            else:
                enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
                if self.gym_feature_attn != 'null': enc_out = enc_out + enc_out_fa

            if self.gym_encoder_only: # encoder-only
                if self.gym_input_embed == 'inverted-encoding':
                    dec_out = self.head(enc_out) # BxDxd_model -> BxDxpred_len
                    dec_out = dec_out.permute(0, 2, 1) # BxDxpred_len -> Bxpred_lenxD
                    dec_out = dec_out[:, :, :D]
                elif self.gym_input_embed == 'series-encoding':
                    if self.series_sampling:
                        dec_out = [self.decoder_projection[i](_) for i, _ in enumerate(enc_out)]
                        dec_out = [self.head[i](_.permute(0, 2, 1)).permute(0, 2, 1) for i, _ in enumerate(dec_out)]
                        dec_out = [_[:, -self.pred_len:, :] for _ in dec_out]
                        dec_out = torch.stack(dec_out, dim=-1).sum(-1)
                    else:
                        dec_out = self.decoder_projection(enc_out) # BxSxd_model -> BxSxD
                        # projection to predicting length
                        dec_out = self.head(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # BxSxD -> BxDxS -> BxDxpred_len -> Bxpred_lenxD
                        dec_out = dec_out[:, -self.pred_len:, :]
                    if self.gym_channel_independent: dec_out = dec_out.reshape(B, D, self.pred_len).permute(0, 2, 1).contiguous()
                elif self.gym_input_embed == 'series-patching':
                    if self.series_sampling:
                        enc_out = [torch.reshape(_, (-1, n_vars, _.shape[-2], _.shape[-1])) for _ in enc_out]
                        enc_out = [_.permute(0, 1, 3, 2) for _ in enc_out]
                        dec_out = [self.head[i](_) for i, _ in enumerate(enc_out)]
                        dec_out = [_.permute(0, 2, 1) for _ in dec_out]
                        dec_out = torch.stack(dec_out, dim=-1).sum(-1)                   
                    else:
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
                if self.gym_input_embed == 'series-encoding':
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
                if self.gym_input_embed == 'series-encoding':
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
        # 如果混合粒度情况, 用第一(0)层参照TimeMixer
        dec_out = self.series_norm[0](dec_out, 'denorm') if self.series_sampling else self.series_norm(dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
