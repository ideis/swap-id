import os
import glob
import random
from math import cos, pi
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from apex import amp
from model.discriminator import Discriminator

class DiscriminatorTrainer(nn.Module):
    def __init__(self, model_dir, optimizer, lr, num_classes):
        super().__init__()
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.logs_dir = os.path.join(self.model_dir, 'logs')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.writer = SummaryWriter(self.logs_dir)
        self.net = Discriminator(num_classes).cuda()
        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr
        self.optimizer = optimizer([
            {'params': self.net.backbone[:17].parameters(), 'weight_norm': 5e-4},
            {'params': self.net.backbone[17:].parameters(), 'weight_norm': 5e-3},
            {'params': self.net.metric.parameters()}
        ], lr=lr, momentum=0.9)

        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O1")

        self._iter = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.max_iters = 100000
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.cuda()

    @property
    def iter(self):
        return self._iter.item()

    @property
    def device(self):
        return next(self.parameters()).device

    def adapt(self, args):
        device = self.device
        return [arg.to(device) for arg in args]

    def train_loop(self, dataloaders, eval_every):
        for batch in tqdm(dataloaders['train']):
            torch.Tensor.add_(self._iter, 1)
            self.adjust_lr()
            loss = self.train_step(self.adapt(batch))
            stats = self.get_opt_stats()
            self.write_logs(loss=loss, stats=stats)
            if (self.iter % eval_every == 0):
                metrics = self.evaluate(dataloaders)
                self.write_logs(metrics=metrics)
                self.save_checkpoint()

    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.loss(*batch)
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, x, labels):
        logits = self.net(x, labels)
        return self.criterion(logits, labels)

    def evaluate(self, dataloaders):
        self.net.eval()
        with torch.no_grad():
            emb = []
            for x in tqdm(dataloaders['val']):
                x = x.cuda()
                emb.append(self.net(x).cpu())
            emb = torch.cat(emb, 0)
            emb = F.normalize(emb)

            y_hat = (emb[0::2,:] * emb[1::2,:]).sum(1).numpy()
            y_true = np.array(dataloaders['val'].dataset.labels)
            
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_hat)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            acc = metrics.accuracy_score(y_true, y_hat > optimal_threshold)
        self.net.train()
        val_metrics = {
            'Acc': acc
        }
        return val_metrics

    def get_opt_stats(self):
        stats = {'lr' : self.optimizer.param_groups[0]['lr']}
        return stats

    def adjust_lr(self, warmup=10000):
        if self.iter <= warmup:
            lr = self.lr * self.iter / warmup 
        else:
            lr = self.lr * (1 + cos(pi * (self.iter - warmup) / (self.max_iters - warmup))) / 2
        
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def write_logs(self, loss=None, metrics=None, stats=None):
        if loss:
            self.writer.add_scalar('loss/train', loss, self.iter)
        if metrics:
            for name, value in metrics.items():
                self.writer.add_scalar(f'metric/{name}', value, self.iter)
        if stats:
            for name, value in stats.items():
                self.writer.add_scalar(f'stats/{name}', value, self.iter)

    def save_checkpoint(self, max_checkpoints=100):
        checkpoints = glob.glob(f'{self.model_dir}/*.pt')
        if len(checkpoints) > max_checkpoints:
            os.remove(checkpoints[-1])
        with open(f'{self.model_dir}/{self.iter}.pt', 'wb') as f:
            torch.save(self.net.state_dict(), f)

    def load_checkpoint(self, path, load_last=True):
        if load_last:
            try:
                checkpoints = glob.glob(f'{path}/*.pt')
                path = max(checkpoints, key=os.path.getctime)
            except (ValueError):
                print(f'Directory is empty: {path}')

        try:
            self.net.load_state_dict(torch.load(path), strict=False)
            iter_str = ''.join(filter(lambda x: x.isdigit(), path))
            self._iter = nn.Parameter(torch.tensor(int(iter_str)), requires_grad=False)
            self.cuda()
        except (FileNotFoundError):
            print(f'No such file: {path}')

    def release(self):
        self.net.eval()
        if not os.path.exists(os.path.join(self.model_dir, 'release')):
            os.makedirs(os.path.join(self.model_dir, 'release'))
        torch.jit.script(self.net.cpu()).save(os.path.join(self.model_dir, f'release/model_{self.iter}.pt'))
        self.net.train()
