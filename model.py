import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

########################################################################################################
# MHA: Multi-head Attention + Rotary Encoding
########################################################################################################

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)
@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[...,:q.shape[-2],:], sin[...,:q.shape[-2],:]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)         # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)       # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                                     # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))                 # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))                     # causal mask
        att = F.softmax(att, dim = -1)                                                  # softmax

        x = att @ v                                                                     # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)                               # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x

class GeGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_sz = 3 * config.n_ffn
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        v = self.value(x)        
        y = self.weight(F.gelu(k) * v)
        return y

########################################################################################################
# The GPT Model with our blocks
########################################################################################################

class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k,v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if self.config.USE_SMALL_EMB and self.layer_id == 0: # LN(SmallInit(Emb))
            self.lnPre = nn.LayerNorm(config.n_embd)
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.att = MHA(config)
        self.ffn = GeGLU(config)

    def forward(self, x):
        if self.config.USE_SMALL_EMB and self.layer_id == 0: # LN(SmallInit(Emb))
            x = self.lnPre(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ctx_len = config.ctx_len

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):

        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)        

        if isinstance(module, (nn.Embedding)):
            if self.config.USE_SMALL_EMB:
                nn.init.uniform_(module.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        decay = set() # separate out all parameters to those that will and won't experience regularizing weight decay
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias') or ('time' in fpn) or ('head' in fpn) or ('scale' in fpn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()} # validate that we considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(idx)

        x = self.blocks(x)

        x = self.ln_out(x)

        x = self.head(x)
        
        # print(self.emb.weight.detach().cpu().numpy()) # <-------- Show embedding matrix changes
                
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss
