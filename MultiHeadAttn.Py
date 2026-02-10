import torch
import torch.nn as nn

class MultiHeadAttn(nn.Module):
    def __init__(self, d_in,d_out, context_len, dropout, num_heads, qkv_bias=False):
      super().__init__()
      assert(d_out%num_heads==0)
      self.d_out=d_out
      self.num_heads=num_heads
      self.head_dim=d_out//num_heads

      self.W_query=nn.Linear(d_in,d_out,bias=qkv_bias)
      self.W_key=nn.Linear(d_in,d_out,bias=qkv_bias)
      self.W_value=nn.Linear(d_in,d_out,bias=qkv_bias)

      self.out_projection=nn.Linear(d_out,d_in)
      self.dropout=nn.Dropout(dropout)
      self.register_buffer("mask", torch.triu(torch.ones(context_len,context_len), diagonal=1))

    def forward(self,x):
        b, num_tokens, d_in= x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys= keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries= queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values=values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)

        attn_scores=queries@keys.transpose(2,3)
        mask_bool=self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights=self.dropout(attn_weights)



        context_vecs=(attn_weights@values).transpose(1,2)
        context_vecs = context_vecs.contiguous().view(b,num_tokens,self.d_out)
        context_vecs = self.out_projection(context_vecs)

        return context_vecs



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

class Generate_Text():
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