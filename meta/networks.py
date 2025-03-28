import torch
from torch import nn
import inspect

# meta classifier by using MetaOD feature extractor
# change the embedding unlearnable?
class meta_predictor(nn.Module):
    def __init__(self, n_col, embed_dim_meta_feature=156, d_model=64, dropout=0.1):
        super(meta_predictor, self).__init__()
        embedding_dim = embed_dim_meta_feature // len(n_col)
        self.embeddings = nn.ModuleList([nn.Embedding(int(_), embedding_dim) for _ in n_col])
        embed_dim_component = len(n_col) * embedding_dim
        print(f"embed_dim_component (total): {embed_dim_component}, embed_dim_meta_feature: {embed_dim_meta_feature}")
        
        self.out = nn.Sequential(nn.Linear(embed_dim_component + embed_dim_meta_feature, d_model),
                                 nn.LayerNorm(d_model),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(d_model, d_model),
                                 nn.LayerNorm(d_model),
                                 nn.LeakyReLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(d_model, 1))
        
        # from nanoGPT: https://github.com/karpathy/nanoGPT/blob/eba36e86449f3c56d840a93092cb779a260544d08/model.py#L263
    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), model=None):
        # start with all of the candidate parameters
        if model is not None:
            param_dict = {pn: p for pn, p in model.named_parameters()}
        else:
            param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        # fused版本优化器在GPU上运算效率更高
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extras_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extras_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def forward(self, components, meta_feature):
        component_embedding = torch.hstack([e(components[:, i]) for i, e in enumerate(self.embeddings)])
        embedding = torch.cat((component_embedding, meta_feature), dim=1)
        pred = self.out(embedding)

        return component_embedding, pred