# TSGym

## Design Dimensions

### Data Augmentation (todo)
- tslib

### ↓ Series Normalization
- None
- Statistic
- RevIN
- DishTS

### ↓ Series Decomposition
- None
- Moving Average
- MoE Moving Average (from FEDformer)
- DFT
- multi-resolution (like TimesNet and TimeMixer++, todo)

### ↓ Series Embedding (tokenization)
- channel-dependent
    - series-encoding (linear projection + positional-encoding for transformer-based; linear projection for non-transformer-based)
    - inverted-encoding (in iTransformer)
- channel-independent
    - series-encoding
    - series-patching
- channel-attention (todo)

### ↓ Series Mixing
- seasonal mixing (high -> low) & trend mixing (low -> high)
    - cross-attention
    - linear projection (like Timixer, todo) 

### ↓ Network Architecture
- MLP
    - TSMixer (todo)
- GRU
    - segRNN (alignment, todo)
- CNN (e.g., TimesNet)
- Transformer
    - Self-attention
    - Auto-Correlation
    - Sparse Attention
    - Frequency Enhanced Attention
    - TwostageAttention (todo)
    - Nonstationary Attention
- LLM
    - GPT4TS
    - TimeLLM
- TSFM
    - Timer
    - Moment(-base)
    
ps: w.r.t. LLM and TSFM, series-patching is the default option.

## todo
- <del>20241211: TransformerGym_None_series-patching_sparse-attention, loss全是0? (nan)</del>
- inverse attention
- <del>multi-resolution (cross-attention based)</del>
- <del>channel-independent</del>
- decoder structure
- [iTransformer中的截断问题](https://github.com/thuml/Time-Series-Library/blob/cdf8f0c3c5e79c1e8152e71dc35009ae46a6a920/models/iTransformer.py#L101C60-L101C70)
- seasonal/trend用不同的enc_embedding?
- TwoStageAttention layer?
- MLP-based multi-resolution mixing
- data augmentation
- bug: seq_len=192和series-patching
- 有些module无法组合: 例如inverted-encoding + series-patching
- x_mark的有效性