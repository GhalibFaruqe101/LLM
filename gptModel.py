# %% [markdown]
# ![image.png](attachment:image.png)
# - Configuration details for the 124 million parameter GPT-2 model 

# %%
import _frozen_importlib_external
import sysconfig
from importlib.metadata import version


# %%
import torch
import torch.nn as nn


# %%
GPT_CONFIG_24M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_head":12,
    "n_layer":12,
    "drop_rate":0.1,
    "qkv_bias": False
}

# %%
import tiktoken 
tokenizer = tiktoken.get_encoding("gpt2")

# %% [markdown]
# ![image.png](attachment:image.png)
# 
# - normalization is the activitions nn layer around mean of 0 and normalize variance to 1
# - helps to stabilize and faster convergences to effective weights
# - applied both before and after the multi-head attention, final output layer
# 
# ![image-2.png](attachment:image-2.png)
# 
# - dimension [ dim--> 0 the first index
#               dim--> -1 the last index]
# - scale and shifts are trainable parameters that helps to auto adjust during training
# - eps (~ 0) to avoid division-to-zero error incase the var is 0

# %%
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# %% [markdown]
# ![image.png](attachment:image.png)
# - GELU is non-linear function, non-zero gradient for negative values

# %%
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3))))

# %% [markdown]
# ![image.png](attachment:image.png)

# %%


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)



# %%
ffn = FeedForward(GPT_CONFIG_24M)

# input shape: [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768) 
out = ffn(x)
print(out.shape)

# %% [markdown]
# ![image.png](attachment:image.png)
# - shortcut or skip/residual connection to solve the gradient problem
# - it works by connecting the output of one layer to another 

# %%
from MultiHeadAttn import MultiHeadAttn



# %%

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn=MultiHeadAttn(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_len=cfg["context_length"],
            num_heads=cfg[ "n_head"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop_shortcut= nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x= self.drop_shortcut(x)
        x = x + shortcut

        return x

# %%
torch.manual_seed(123)

x = torch.rand(2, 4, 768) 
block = TransformerBlock(GPT_CONFIG_24M )
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)


# %% [markdown]
# ![image.png](attachment:image.png)

# %%


# %%
import torch
import torch.nn as nn


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

     
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

       
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layer"])]
        )

       
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_indx):
       
        if not isinstance(in_indx, torch.Tensor):
            in_indx = torch.tensor(in_indx, dtype=torch.long)

        batch_size, seq_len = in_indx.shape

    
        token_embedding = self.tok_emb(in_indx)

        
        pos = torch.arange(seq_len, device=in_indx.device)
        position_embedding = self.pos_emb(pos)

     
        x = token_embedding + position_embedding

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits


# %%

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_24M)
batch = []

print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# %%
def generate_text (model, idx, max_token, context_size):
    for _ in range(max_token):
        idx_condition = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_condition)
            logits = logits[:,-1,:]
            probability = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probability,dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# %%


# %% [markdown]
# ![image.png](attachment:image.png)
# 
# - 'unsqueeze' is used to add dimension. if dim[T]-->dim[B T]

# %%

starting_context = "Hello, I am"
encode = tokenizer.encode(starting_context)
encode_tensor = torch.tensor(encode).unsqueeze(0)

# %%
model.eval()
out = generate_text (model=model, idx=encode_tensor, max_token=6, context_size=GPT_CONFIG_24M["context_length"])


# %%
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)


# %% [markdown]
# | Keyword          | What it Does                | Input Shape | Output Shape | Why Used in GPT          |
# | ---------------- | --------------------------- | ----------- | ------------ | ------------------------ |
# | `unsqueeze(dim)` | Adds a dimension of size 1  | `[T]`       | `[1, T]`     | Add batch dimension      |
# | `squeeze(dim)`   | Removes dimension of size 1 | `[1, T]`    | `[T]`        | Remove fake dimensions   |
# | `view()`         | Reshapes tensor             | `[B, T, C]` | custom       | Split heads in attention |
# | `reshape()`      | Safe reshape                | `[B, T, C]` | custom       | Same as view but safer   |
# | `transpose(a,b)` | Swaps dimensions            | `[B,T,C]`   | `[B,C,T]`    | Attention math           |
# | `permute()`      | Reorders dimensions         | `[B,T,H,D]` | `[B,H,T,D]`  | Multi-head attention     |
# | `expand()`       | Broadcast without copy      | `[1,T,C]`   | `[B,T,C]`    | Position embeddings      |
# | `repeat()`       | Copies data                 | `[1,T,C]`   | `[B,T,C]`    | Rare in GPT              |
# 

# %% [markdown]
# | Keyword           | What it Does             | Where Used                 |
# | ----------------- | ------------------------ | -------------------------- |
# | `nn.Module`       | Base class for models    | GPTModel, TransformerBlock |
# | `__init__()`      | Defines layers           | Model construction         |
# | `forward()`       | Defines computation      | Model execution            |
# | `model(x)`        | Calls forward internally | Training & inference       |
# | `model.eval()`    | Disable dropout          | Generation                 |
# | `torch.no_grad()` | Disable gradients        | Inference                  |
# 

# %% [markdown]
# | Component            | Purpose             | Input     | Output    |
# | -------------------- | ------------------- | --------- | --------- |
# | `nn.Embedding`       | Token lookup        | `[B,T]`   | `[B,T,C]` |
# | Token Embedding      | Word representation | token ids | vectors   |
# | Positional Embedding | Position info       | positions | vectors   |
# | TransformerBlock     | Attention + FFN     | `[B,T,C]` | `[B,T,C]` |
# | LayerNorm            | Stabilizes training | `[B,T,C]` | `[B,T,C]` |
# | Linear               | Projection          | `[B,T,C]` | `[B,T,V]` |
# 

# %% [markdown]
# | Keyword            | Purpose                  | Example              |
# | ------------------ | ------------------------ | -------------------- |
# | `logits[:, -1, :]` | Last token prediction    | next token           |
# | `argmax()`         | Greedy sampling          | deterministic output |
# | `softmax()`        | Convert to probabilities | sampling             |
# | `torch.cat()`      | Append token             | grow sequence        |
# 

