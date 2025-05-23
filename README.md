# TSGym

## Design Dimensions

### Data Augmentation
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

### Network Training