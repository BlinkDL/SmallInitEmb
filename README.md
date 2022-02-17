# SmallInitEmb
LayerNorm(SmallInit(Embedding)) in a Transformer

I find that when training a transformer, the embedding matrix moves slowly, hence it's difficult for the model to jump out of the initial noisy embedding.
```
(initial embedding)
[[-0.0073  0.0062 -0.0261 ...  0.0086  0.0107 -0.008 ] ... ]
 (after 1 step, the directions of the embedding vectors are not moved much because the numbers change by ~LR = ~4e-4)
[[-0.0069  0.0066 -0.0265 ...  0.009   0.0111 -0.0084] ... ]
```
So I propose initializing the embedding matrix to tiny values, and put another LayerNorm after it (before all the SA & FFN layers):
```
if isinstance(module, (nn.Embedding)):
    nn.init.uniform_(module.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
...
if self.config.USE_SMALL_EMB and self.layer_id == 0:
    x = self.lnPre(x) # LN(SmallInit(Emb))
x = x + self.att(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```
And then you get improved convergence (especially for BPE models) because the model can quickly jump out of the tiny initial embedding (small changes after 1 step -> significant changes of directions -> significant changes after LayerNorm).

Loss curve comparison: https://wandb.ai/blinkdl/SmallEmbTest

(the gap between LayerNorm(SmallEmb)) and baseline persists after more training)

# Moreover, you can directly train PostLN models without warmup with SmallInit(Emb)
```
if isinstance(module, (nn.Embedding)):
    nn.init.uniform_(module.weight, a=-1e-4, b=1e-4) # SmallInit(Emb)
...
x = self.ln1(x) # this plays the same role as the lnPre in the above PreLN code
x = x + self.att(x)
x = self.ln2(x)
x = x + self.ffn(x)
(note you shall have another LN after the final ffn)
```
