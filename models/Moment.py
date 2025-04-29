# https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb

import torch
from torch import nn
from momentfm import MOMENTPipeline

# todo: https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/forecasting.ipynb
# 注意原始Moment代码里面有mixed precision training, scheduler, gradient clipping
class Model(nn.Module):
    def __init__(self, frozen=True):
        super(Model, self).__init__()
        self.frozen = frozen
        self.model_pretrained = self._build_model()

    def _build_model(self):
        model = MOMENTPipeline.from_pretrained(
            f"./models/llm/MOMENT-base", 
            model_kwargs={
                'task_name': 'forecasting',
                'forecast_horizon': 96, # pred_len
                'head_dropout': 0.1,
                'weight_decay': 0,
                'freeze_encoder': self.frozen, # Freeze the patch embedding layer (default: True)
                'freeze_embedder': self.frozen, # Freeze the transformer encoder (default: True)
                'freeze_head': False, # The linear forecasting head must be trained
            },
            local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
        )

        model.init()

        # only extract the encoder part
        self.block = model.encoder.block
        self.final_layer_norm = model.encoder.final_layer_norm
        self.dropout = model.encoder.dropout

        return model
    
    # def forward(self, x, input_mask=None):
    #     # the input shape of Moment should be: batch size x n_channels x context_length
    #     S = x.shape[1] # sequence length
    #     if S < 512: # BxSxD
    #         # print(f'the shape of org x: {x.shape}')
    #         x = torch.cat((x, torch.zeros(x.shape[0], 512 - S, x.shape[2]).to(x.device)), dim=1)
    #         # print(f'the shape of concat x: {x.shape}')
    #         input_mask = torch.ones(x.shape[0], 512).to(x.device)
    #         input_mask[:, S:] = 0.0

    #     x = self.model_pretrained(x_enc=x.permute((0, 2, 1)), input_mask=input_mask) # BxSxD -> BxDxS
    #     return x

    def forward(self, x):
        for layer in self.block:
            x = layer(x)[0]  # 只取hidden_states
        x = self.final_layer_norm(x)
        x = self.dropout(x)
        return x, None