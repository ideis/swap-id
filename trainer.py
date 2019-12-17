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
        self.generator_mae_weight = 0.0001
        self.generator_bce_weight = 0.001
        self.generator_ce_weight = 1
        self.discriminator_bce_real_weight = 0.001
        self.discriminator_bce_fake_weight = 0.001
        self.discriminator_ce_weight = 1

        self.lr = lr
        self.g_optimizer = g_optimizer(self.generator.parameters(), lr=lr, momentum=0.9)
        self.d_optimizer = d_optimizer(self.discriminator.parameters(), lr=lr, momentum=0.9)

        (self.generator, self.discriminator), (self.g_optimizer, self.d_optimizer) = amp.initialize([self.generator, self.discriminator],
                                                                                                    [self.g_optimizer, self.d_optimizer],
                                                                                                    opt_level="O1",
                                                                                                    num_losses=6)

        self._iter = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.max_iters = 100000

        if torch.cuda.is_available():
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

    def train_loop(self, dataloaders, eval_every, generate_every, save_every):
        for batch in tqdm(dataloaders['train']):
            torch.Tensor.add_(self._iter, 1)
            # generator step
            if self._iter % 2 == 0:
                self.adjust_lr(self.g_optimizer)
                g_losses = self.g_step(self.adapt(batch))
                g_stats = self.get_opt_stats(self.g_optimizer, type='generator')
                self.write_logs(losses=g_losses, stats=g_stats, type='generator')

            #discriminator step
            if self._iter % 2 == 1:
                self.adjust_lr(self.d_optimizer)
                d_losses = self.d_step(self.adapt(batch))
                d_stats = self.get_opt_stats(self.d_optimizer, type='discriminator')
                self.write_logs(losses=d_losses, stats=d_stats, type='discriminator')

            if self.iter % eval_every == 0:
                x, labels = batch
                x = x.cuda()
                metrics = self.evaluate(x)
                self.write_logs(metrics=metrics)
                metrics = self.evaluate_identification(dataloaders)
                self.write_logs(metrics=metrics)


            if self.iter % generate_every == 0:
                x, labels = batch
                x = x[:2, ...].cuda()
                self.generate(x)

            if self.iter % save_every == 0:
                self.save_discriminator()
                self.save_generator()


    def g_step(self, batch):
        self.g_optimizer.zero_grad()
        mae_loss, bce_loss, ce_loss = self.g_loss(*batch)
        with amp.scale_loss(mae_loss * self.generator_mae_weight, self.g_optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(bce_loss * self.generator_bce_weight, self.g_optimizer, loss_id=0) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(ce_loss * self.generator_ce_weight, self.g_optimizer, loss_id=1) as scaled_loss:
            scaled_loss.backward()
        self.g_optimizer.step()

        losses = {
            'mae': self.generator_mae_weight * mae_loss.item(),
            'bce_fake': self.generator_bce_weight * bce_loss.item(),
            'ce':  self.generator_ce_weight * ce_loss.item()
        }
        return losses

    def d_step(self, batch):
        self.d_optimizer.zero_grad()
        bce_fake, bce_real, ce_loss = self.d_loss(*batch)
        with amp.scale_loss(bce_fake * self.discriminator_bce_fake_weight, self.d_optimizer, loss_id=2) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(bce_real * self.discriminator_bce_real_weight, self.d_optimizer, loss_id=3) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        with amp.scale_loss(ce_loss * self.discriminator_ce_weight, self.d_optimizer, loss_id=4) as scaled_loss:
            scaled_loss.backward()
        self.d_optimizer.step()

        losses = {
            'bce_real': self.discriminator_bce_real_weight * bce_real.item(),
            'bce_fake': self.discriminator_bce_fake_weight * bce_fake.item(),
            'ce': self.discriminator_ce_weight * ce_loss.item()
        }
        return losses

    def g_loss(self, x, labels):
        src, tgt = torch.chunk(x, 2, dim=0)
        src_label, tgt_label = torch.chunk(labels, 2)

        fake = self.generator(src, tgt)
        mae_loss = self.mae_criterion(fake, src)

        logits, fake_realness_logit  = self.discriminator(fake, tgt_label)
        bce_loss = self.bce_criterion(fake_realness_logit, torch.ones_like(fake_realness_logit) - random.uniform(0.1, 0.3))
        ce_loss = self.ce_criterion(logits, tgt_label)
        return mae_loss, bce_loss, ce_loss
    
    def d_loss(self, x, labels):
        src, tgt = torch.chunk(x, 2, dim=0)
        with torch.no_grad():
            fake = self.generator(src, tgt)

        logits, real_realness_logit  = self.discriminator(x, labels)
        ce_loss = self.ce_criterion(logits, labels)

        _, fake_realness_logit = self.discriminator(fake)

        bce_fake = self.bce_criterion(fake_realness_logit, torch.zeros_like(fake_realness_logit) + random.uniform(0.1, 0.3))
        bce_real = self.bce_criterion(real_realness_logit, torch.ones_like(real_realness_logit) - random.uniform(0.1, 0.3)) / 2
     
        return bce_fake, bce_real, ce_loss

    def evaluate(self, x):
        src, tgt = torch.chunk(x, 2, dim=0)        

        self.generator.eval()
        with torch.no_grad():
            fake = self.generator(src, tgt)
        self.generator.train()

        self.discriminator.eval()
        with torch.no_grad():
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

    
    def evaluate_identification(self, dataloaders):
        self.discriminator.eval()
        with torch.no_grad():
            embs= []
            for x in tqdm(dataloaders['val']):
                x = x.cuda()
                emb, _ = self.discriminator(x)
                emb = emb.cpu()
                embs.append(emb)
            embs = torch.cat(embs, 0)
            embs = F.normalize(embs)

            y_hat = (embs[0::2,:] * embs[1::2,:]).sum(1).numpy()
            y_true = np.array(dataloaders['val'].dataset.labels)
            
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_hat)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            acc = metrics.accuracy_score(y_true, y_hat > optimal_threshold)
        self.discriminator.train()
        val_metrics = {
            'identification_acc': acc
        }
        return val_metrics


    def generate(self, x):
        src, tgt = torch.chunk(x, 2, dim=0)        
        x = torch.cat([src, tgt], dim=1)

        self.generator.eval()
        fake = self.generator(src, tgt)
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

    def adjust_lr(self, optimizer, warmup=5000):
        if self.iter <= warmup:
            lr = self.lr * self.iter / warmup 
        else:
            lr = self.lr * (1 + cos(pi * (self.iter - warmup) / (self.max_iters - warmup))) / 2
        
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr

    def write_logs(self, losses=None, metrics=None, stats=None, type='loss'):
        if losses:
            for name, value in losses.items():
                self.writer.add_scalar(f'{type}/{name}', value, self.iter)
        if metrics:
            for name, value in metrics.items():
                self.writer.add_scalar(f'metric/{name}', value, self.iter)
        if stats:
            for name, value in stats.items():
                self.writer.add_scalar(f'stats/{name}', value, self.iter)

    def save_generator(self, max_checkpoints=100):
        checkpoints = glob.glob(f'{self.model_dir}/*.pt')
        if len(checkpoints) > max_checkpoints:
            os.remove(checkpoints[-1])
        with open(f'checkpoints/{self.model_dir}/generator_{self.iter}.pt', 'wb') as f:
            torch.save(self.generator.state_dict(), f)

    def save_discriminator(self, max_checkpoints=100):
        checkpoints = glob.glob(f'{self.model_dir}/*.pt')
        if len(checkpoints) > max_checkpoints:
            os.remove(checkpoints[-1])
        with open(f'checkpoints/{self.model_dir}/discriminator_{self.iter}.pt', 'wb') as f:
            torch.save(self.discriminator.state_dict(), f)

    def load_discriminator(self, path, load_last=True):
        if load_last:
            try:
                checkpoints = glob.glob(f'{path}/discriminator_*.pt')
                path = max(checkpoints, key=os.path.getctime)
            except (ValueError):
                print(f'Directory is empty: {path}')

        try:
            self.discriminator.load_state_dict(torch.load(path))
            self.cuda()
        except (FileNotFoundError):
            print(f'No such file: {path}')

    def load_generator(self, path, load_last=True):
        if load_last:
            try:
                checkpoints = glob.glob(f'{path}/generator_*.pt')
                path = max(checkpoints, key=os.path.getctime)
            except (ValueError):
                print(f'Directory is empty: {path}')

        try:
            self.generator.load_state_dict(torch.load(path))
            iter_str = ''.join(filter(lambda x: x.isdigit(), path))
            self._iter = nn.Parameter(torch.tensor(int(iter_str)), requires_grad=False)
            self.cuda()
        except (FileNotFoundError):
            print(f'No such file: {path}')
