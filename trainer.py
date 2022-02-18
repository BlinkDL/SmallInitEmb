import math, sys, datetime
import logging
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)

print('logging to wandb... (comment it if you don\'t have wandb)')
import os
import wandb # comment this if you don't have wandb

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    weight_decay = 0.01
    lr_decay = True # cosine decay
    final_tokens = 260e9 # at which point do we reach lr_final
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0 # for DataLoader
    USE_FP16 = False
    GRAD_ACCUM = 1

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.steps = 0

        if 'wandb' in sys.modules:
            cfg = model.config
            for k in config.__dict__:
                setattr(cfg, k, config.__dict__[k]) # combine cfg
            wandb.init(project="SmallEmbTest", name=self.get_run_name() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), config=cfg, save_code=False)

        self.device = 'cpu'
        if torch.cuda.is_available(): # take over whatever gpus are on the system
            self.device = torch.cuda.current_device()

    def get_run_name(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + str(cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        self.now_loss = 0

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.USE_FP16)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset

            loader = DataLoader(data, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)

            pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)
            
            for it, (x, y) in pbar:
                x = x.to(self.device) # place data on the correct device
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    with torch.cuda.amp.autocast(enabled=self.config.USE_FP16):
                        _, loss = model(x, y) # forward the model

                if is_train: # backprop and update the parameters
                    if self.config.GRAD_ACCUM > 1:
                        loss = loss / self.config.GRAD_ACCUM
                        self.now_loss += loss.item()
                    scaler.scale(loss).backward()

                    if (self.steps + 1) % self.config.GRAD_ACCUM == 0:
                        if config.grad_norm_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        model.zero_grad()

                        if config.lr_decay: # decay the learning rate based on our progress
                            self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                            lr_final_factor = config.lr_final / config.learning_rate

                            progress = float(self.tokens) / float(max(1, config.final_tokens))
                            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
                            lr = config.learning_rate * lr_mult

                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr
                        else:
                            lr = config.learning_rate

                        if 'wandb' in sys.modules:
                            wandb.log({"loss": self.now_loss}, step = (self.steps + 1 - self.config.GRAD_ACCUM) * self.config.batch_size)

                        if self.avg_loss < 0:
                            self.avg_loss = self.now_loss
                        else:
                            factor = 1 / (it / self.config.GRAD_ACCUM + 1)
                            self.avg_loss = self.avg_loss * (1.0 - factor) + self.now_loss * factor
                        pbar.set_description(f"epoch {epoch+1} prog {progress*100.0:.2f}% iter {it}: ppl {math.exp(self.avg_loss):.2f} loss {self.avg_loss:.4f} lr {lr:e}")
                        self.now_loss = 0
                    self.steps += 1

        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            
            if (self.config.epoch_save_frequency > 0 and epoch % self.config.epoch_save_frequency == 0) or (epoch == config.max_epochs - 1):
                raw_model = self.model.module if hasattr(self.model, "module") else self.model # DataParallel wrappers keep raw model object in .module
                torch.save(raw_model, self.config.epoch_save_path + str(epoch+1) + '.pth')
