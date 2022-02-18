import os, sys, time, math, random, json, datetime, logging
import numpy as np
import torch
from torch.utils.data import Dataset
from trainer import Trainer, TrainerConfig
from model import GPT, GPTConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

########################################################################################################

USE_SMALL_EMB = True # <-------- The LN(SmallInit(Emb)) trick

USE_FP16 = False    # Mixed Precision?
GRAD_ACCUM = 1      # Gradient accumulation? 1 = disable
batch_size = 16

ctx_len = 256
n_layer = 6
n_head = 8
n_embd = n_head * 64
n_attn = n_embd
n_ffn = n_embd

lr_init = 4e-4
lr_final = 4e-5
n_epoch = 200

betas = (0.9, 0.99)
eps = 1e-8
weight_decay = 0.1
grad_norm_clip = 1.0

epoch_save_frequency = 0
epoch_save_path = 'trained-'
epoch_length_fixed = 10000 # an "epoch" is of fixed length and very short here
num_workers = 0

########################################################################################################
# Load Data
########################################################################################################

datafile = u"enwik8-shift-300.bpe"
datafile_encoding = 'utf-16' # I encoded BPE to utf-16
print('loading data... ' + datafile)

class Dataset(Dataset):
    def __init__(self, data, ctx_len):
        print('building token list...', end=' ')
        unique = sorted(list(set(data)))
        xx = 0
        xxObj = {}
        for u in unique:
            xxObj[xx] = u
            xx += 1
        with open('vocab.json', "w", encoding="utf-16") as vocab_file:
            vocab_file.write(json.dumps(xxObj, ensure_ascii=False))

        data_size, vocab_size = len(data), len(unique)
        print('data has %d tokens, %d unique.' % (data_size, vocab_size))
        self.stoi = { ch:i for i,ch in enumerate(unique) }
        self.itos = { i:ch for i,ch in enumerate(unique) }
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return epoch_length_fixed

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.data) - (self.ctx_len + 1)) # cheat: pick a random spot in dataset
        chunk = self.data[i:i+self.ctx_len+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long, device=torch.device('cuda'))
        y = torch.tensor(dix[1:], dtype=torch.long, device=torch.device('cuda'))
        return x, y

train_dataset = Dataset(open(datafile, "r", encoding=datafile_encoding).read(), ctx_len)

########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_attn=n_attn, n_ffn=n_ffn
    , USE_SMALL_EMB=USE_SMALL_EMB)).cuda()

    print('epoch', n_epoch, 'batchsz', batch_size, 'betas', betas, 'eps', eps, 'wd', weight_decay, 'ctx', ctx_len, 'layer', n_layer, 'head', n_head, 'embd', n_embd, 'attn', n_attn, 'ffn', n_ffn)
    tconf = TrainerConfig(max_epochs=n_epoch, batch_size=batch_size, weight_decay=weight_decay, USE_FP16=USE_FP16, GRAD_ACCUM=GRAD_ACCUM,
                            learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                            final_tokens=n_epoch*len(train_dataset)*ctx_len, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    # torch.save(model, 'trained-' + trainer.get_run_name() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
