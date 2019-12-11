import os
import glob
import random
from math import cos, pi
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.optimizers import FusedAdam

from data.transforms import transform_inv
from model.generator import Generator
from model.discriminator import Discriminator

class Trainer(nn.Module):
    def __init__(self, model_dir, g_optimizer, d_optimizer, lr, num_classes):
        super().__init__()
        self.model_dir = model_dir
        if not os.path.exists(f'checkpoints/{model_dir}'):
            os.makedirs(f'checkpoints/{model_dir}')
        self.logs_dir = f'checkpoints/{model_dir}/logs'
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.writer = SummaryWriter(self.logs_dir)

        self.generator = Generator()
        self.discriminator = Discriminator(num_classes)
        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()

        self.mae_criterion = nn.SmoothL1Loss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.mae_weight = 1
        self.bce_weight = 1
        self.ce_weight = 0.01
        self.bce_real_weight = 1
        self.bce_fake_weight = 1

        self.lr = lr
        self.g_optimizer = g_optimizer(self.generator.parameters(), lr=lr)
        self.d_optimizer = d_optimizer(self.discriminator.parameters(), lr=lr, momentum=0.9)

        (self.generator, self.discriminator), (self.d_optimizer, self.g_optimizer) = amp.initialize([self.generator, self.discriminator],
                                                                                                    [self.g_optimizer, self.d_optimizer],
                                                                                                    opt_level="O1",
                                                                                                    num_losses=5)

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

    def train_loop(self, dataloaders, eval_every, generate_every):
        for batch in tqdm(dataloaders['train']):
            torch.Tensor.add_(self._iter, 1)
            # generator step
            self.adjust_lr(self.g_optimizer)
            g_losses = self.g_step(self.adapt(batch))
            g_stats = self.get_opt_stats(self.g_optimizer, type='generator')
            self.write_logs(losses=g_losses, stats=g_stats)

            #discriminator step
            self.adjust_lr(self.d_optimizer)
            d_losses = self.d_step(self.adapt(batch))
            d_stats = self.get_opt_stats(self.d_optimizer, type='discriminator')
            self.write_logs(losses=d_losses, stats=d_stats)

            if (self.iter % eval_every == 0):
                x, labels = batch
                x = x.cuda()
                metrics = self.evaluate(x)
                self.write_logs(metrics=metrics)
                self.save_checkpoint()

            if (self.iter % generate_every == 0):
                x, labels = batch
                x = x[:2, ...].cuda()
                self.generate(x)

    def g_step(self, batch):
        self.g_optimizer.zero_grad()
        mae_loss, bce_loss, ce_loss = self.g_loss(*batch)
        with amp.scale_loss(mae_loss * self.mae_weight, self.g_optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(bce_loss * self.bce_weight, self.g_optimizer, loss_id=1) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(ce_loss * self.ce_weight, self.g_optimizer, loss_id=2) as scaled_loss:
            scaled_loss.backward()

        self.g_optimizer.step()

        losses = {
            'generator_mae': self.mae_weight * mae_loss.item(),
            'generator_bce': self.bce_weight * bce_loss.item(),
            'generator_ce':  self.ce_weight * ce_loss.item()
        }
        return losses

    def d_step(self, batch):
        self.d_optimizer.zero_grad()
        bce_real, bce_fake = self.d_loss(*batch)
        with amp.scale_loss(bce_real * self.bce_real_weight, self.d_optimizer, loss_id=3) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(bce_fake * self.bce_fake_weight, self.d_optimizer, loss_id=4) as scaled_loss:
            scaled_loss.backward()
        self.d_optimizer.step()

        losses = {
            'discriminator_real': self.bce_real_weight * bce_real.item(),
            'discriminator_fake': self.bce_fake_weight * bce_fake.item()
        }
        return losses

    def g_loss(self, x, labels):
        src, tgt = torch.chunk(x, 2, dim=0)
        src_label, tgt_label = torch.chunk(labels, 2)

        res, mask = self.generator(src, tgt) # res.shape (bs, 3, 112, 112); mask.shape (bs, 1, 112, 112)
        fake = mask * res + (1 - mask) * src
        mae_loss = self.mae_criterion(fake, src)

        logits, fake_realness_logit  = self.discriminator(fake, tgt_label)
        bce_loss = self.bce_criterion(fake_realness_logit, torch.ones_like(fake_realness_logit) - random.uniform(0.1, 0.3))
        ce_loss = self.ce_criterion(logits, tgt_label)
        return mae_loss, bce_loss, ce_loss
    
    def d_loss(self, x, labels):
        src, tgt = torch.chunk(x, 2, dim=0)
        src_label, tgt_label = torch.chunk(labels, 2)
        with torch.no_grad():
            res, mask = self.generator(src, tgt) # res.shape (bs, 3, 112, 112); mask.shape (bs, 1, 112, 112)
            fake = mask * res + (1 - mask) * src
    
        _, real_realness_logit = self.discriminator(x)
        _, fake_realness_logit = self.discriminator(fake)

        bce_real = self.bce_criterion(real_realness_logit, torch.ones_like(real_realness_logit) - random.uniform(0.1, 0.3)) / 2
        bce_fake = self.bce_criterion(fake_realness_logit, torch.zeros_like(fake_realness_logit) + random.uniform(0.1, 0.3))
        return bce_real, bce_fake

    def evaluate(self, x):
        src, tgt = torch.chunk(x, 2, dim=0)        

        self.generator.eval()
        res, mask = self.generator(src, tgt)
        fake = mask * res + (1 - mask) * src
        self.generator.train()

        self.discriminator.eval()
        src_emb, _  = self.discriminator(src)
        tgt_emb, _  = self.discriminator(tgt)
        fake_emb, _ = self.discriminator(fake)
        self.discriminator.train()

        src_emb = F.normalize(src_emb)
        tgt_emb = F.normalize(tgt_emb)
        fake_emb = F.normalize(fake_emb)

        src_fake_cos_distance = (src_emb * fake_emb).sum(dim=-1)
        src_fake_angle = torch.acos(src_fake_cos_distance) * (180/pi)

        tgt_fake_cos_distance = (tgt_emb * fake_emb).sum(dim=-1)
        tgt_fake_angle = torch.acos(tgt_fake_cos_distance) * (180/pi)

        _, fake_realness_logit = self.discriminator(fake)
        fake_acc = torch.mean((fake_realness_logit < 0).float())

        _, real_realness_logit = self.discriminator(x)
        real_acc = torch.mean((real_realness_logit > 0).float())

        metrics = {
            'src_fake_angle': src_fake_angle.mean().item(),
            'tgt_fake_angle': tgt_fake_angle.mean().item(),
            'fake_acc': fake_acc.item(),
            'real_acc': real_acc.item()
        }
        return metrics


    def generate(self, x):
        src, tgt = torch.chunk(x, 2, dim=0)        
        x = torch.cat([src, tgt], dim=1)

        self.generator.eval()
        res, mask = self.generator(src, tgt)
        fake = mask * res + (1 - mask) * src
        self.generator.train()

        img_tensors = torch.cat([src, fake, tgt], dim=3)
        img_tensors = img_tensors.squeeze(0).cpu()
        imgs = transform_inv(img_tensors)

        if not os.path.exists(f'results/{self.model_dir}'):
            os.makedirs(f'results/{self.model_dir}')
        imgs.save(f'results/{self.model_dir}/{self.iter}.png')


    def get_opt_stats(self, optimizer, type=''):
        stats = {f'{type}_lr' : optimizer.param_groups[0]['lr']}
        return stats

    def adjust_lr(self, optimizer, warmup=10000):
        if self.iter <= warmup:
            lr = self.lr * self.iter / warmup 
        else:
            lr = self.lr * (1 + cos(pi * (self.iter - warmup) / (self.max_iters - warmup))) / 2
        
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr

    def write_logs(self, losses=None, metrics=None, stats=None):
        if losses:
            for name, value in losses.items():
                self.writer.add_scalar(f'loss/{name}', value, self.iter)
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
        with open(f'checkpoints/{self.model_dir}/{self.iter}.pt', 'wb') as f:
            torch.save(self.generator.state_dict(), f)

    def load_discriminator(self, path, load_last=True):
        if load_last:
            try:
                checkpoints = glob.glob(f'{path}/*.pt')
                path = max(checkpoints, key=os.path.getctime)
            except (ValueError):
                print(f'Directory is empty: {path}')

        try:
            self.discriminator.load_state_dict(torch.load(path))
            self.discriminator.cuda()
        except (FileNotFoundError):
            print(f'No such file: {path}')

    def load_generator(self, path, load_last=True):
        if load_last:
            try:
                checkpoints = glob.glob(f'{path}/*.pt')
                path = max(checkpoints, key=os.path.getctime)
            except (ValueError):
                print(f'Directory is empty: {path}')

        try:
            self.generator.load_state_dict(torch.load(path))
            iter_str = ''.join(filter(lambda x: x.isdigit(), path))
            self._iter = nn.Parameter(torch.tensor(int(iter_str)), requires_grad=False)
            self.generator.cuda()
        except (FileNotFoundError):
            print(f'No such file: {path}')
