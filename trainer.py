import cv2
import os
import glob
import random
from math import cos, pi
from tqdm import tqdm
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from apex import amp

from model.encoder.identity import ArcFaceNet, MobileFaceNet
from model.generator import AEI_Net
from model.discriminator import MultiscaleDiscriminator
from loss import hinge_loss


class Trainer(nn.Module):
    def __init__(self, model_dir, g_optimizer, d_optimizer, lr, warmup, max_iters):
        super().__init__()
        self.model_dir = model_dir
        if not os.path.exists(f'checkpoints/{model_dir}'):
            os.makedirs(f'checkpoints/{model_dir}')
        self.logs_dir = f'checkpoints/{model_dir}/logs'
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.writer = SummaryWriter(self.logs_dir)

        self.arcface = ArcFaceNet(50, 0.6, 'ir_se').cuda()
        self.arcface.eval()
        self.arcface.load_state_dict(torch.load('checkpoints/model_ir_se50.pth', map_location='cuda'), strict=False)

        self.mobiface = MobileFaceNet(512).cuda()
        self.mobiface.eval()
        self.mobiface.load_state_dict(torch.load('checkpoints/mobilefacenet.pth', map_location='cuda'), strict=False)

        self.generator = AEI_Net().cuda()
        self.discriminator = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).cuda()

        self.adversarial_weight = 1
        self.src_id_weight = 5
        self.tgt_id_weight = 1
        self.attributes_weight = 10
        self.reconstruction_weight = 10

        self.lr = lr
        self.warmup = warmup
        self.g_optimizer = g_optimizer(self.generator.parameters(), lr=lr, betas=(0, 0.999))
        self.d_optimizer = d_optimizer(self.discriminator.parameters(), lr=lr, betas=(0, 0.999))

        self.generator, self.g_optimizer = amp.initialize(self.generator, self.g_optimizer, opt_level="O1")
        self.discriminator, self.d_optimizer = amp.initialize(self.discriminator, self.d_optimizer, opt_level="O1")

        self._iter = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.max_iters = max_iters

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
            # if self.iter % 2 == 0:
            # self.adjust_lr(self.g_optimizer)
            g_losses = self.g_step(self.adapt(batch))
            g_stats = self.get_opt_stats(self.g_optimizer, type='generator')
            self.write_logs(losses=g_losses, stats=g_stats, type='generator')

            # #discriminator step
            # if self.iter % 2 == 1:
            # self.adjust_lr(self.d_optimizer)
            d_losses = self.d_step(self.adapt(batch))
            d_stats = self.get_opt_stats(self.d_optimizer, type='discriminator')
            self.write_logs(losses=d_losses, stats=d_stats, type='discriminator')

            if self.iter % eval_every == 0:
                discriminator_acc = self.evaluate_discriminator_accuracy(dataloaders['val'])
                identification_acc = self.evaluate_identification_similarity(dataloaders['val'])
                metrics = {**discriminator_acc, **identification_acc}
                self.write_logs(metrics=metrics)

            if self.iter % generate_every == 0:
                self.generate(*self.adapt(batch))

            if self.iter % save_every == 0:
                self.save_discriminator()
                self.save_generator()

    def g_step(self, batch):
        self.generator.train()
        self.g_optimizer.zero_grad()
        L_adv, L_src_id, L_tgt_id, L_attr, L_rec, L_generator = self.g_loss(*batch)
        with amp.scale_loss(L_generator, self.g_optimizer) as scaled_loss:
            scaled_loss.backward()
        self.g_optimizer.step()

        losses = {
            'adv': L_adv.item(),
            'src_id':  L_src_id.item(),
            'tgt_id':  L_tgt_id.item(),
            'attributes':  L_attr.item(),
            'reconstruction': L_rec.item(),
            'total_loss': L_generator.item()
        }
        return losses

    def d_step(self, batch):
        self.discriminator.train()
        self.d_optimizer.zero_grad()
        L_fake, L_real, L_discriminator = self.d_loss(*batch)
        with amp.scale_loss(L_discriminator, self.d_optimizer) as scaled_loss:
            scaled_loss.backward()
        self.d_optimizer.step()

        losses = {
            'hinge_fake': L_fake.item(),
            'hinge_real': L_real.item(),
            'total_loss': L_discriminator.item()
        }
        return losses

    def g_loss(self, Xs, Xt, same_person):
        with torch.no_grad():
            src_embed = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            tgt_embed = self.arcface(F.interpolate(Xt[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))

        Y_hat, Xt_attr = self.generator(Xt, src_embed, return_attributes=True)        

        Di = self.discriminator(Y_hat)

        L_adv = 0
        for di in Di:
            L_adv += hinge_loss(di[0], True)

        fake_embed = self.arcface(F.interpolate(Y_hat[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        L_src_id =(1 - torch.cosine_similarity(src_embed, fake_embed, dim=1)).mean()
        L_tgt_id =(1 - torch.cosine_similarity(tgt_embed, fake_embed, dim=1)).mean()

        batch_size = Xs.shape[0]
        Y_hat_attr = self.generator.get_attr(Y_hat)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_hat_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y_hat - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
        L_generator = (self.adversarial_weight * L_adv) + (self.src_id_weight * L_src_id) + (self.tgt_id_weight * L_tgt_id) + (self.attributes_weight * L_attr) + (self.reconstruction_weight * L_rec)
        return L_adv, L_src_id, L_tgt_id, L_attr, L_rec, L_generator
    
    def d_loss(self, Xs, Xt, same_person):
        with torch.no_grad():
            src_embed = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        Y_hat = self.generator(Xt, src_embed, return_attributes=False)
          
        fake_D = self.discriminator(Y_hat.detach())
        L_fake = 0
        for di in fake_D:
            L_fake += hinge_loss(di[0], False)
        real_D = self.discriminator(Xs)
        L_real = 0
        for di in real_D:
            L_real += hinge_loss(di[0], True)

        L_discriminator = 0.5*(L_real + L_fake)
        return L_fake, L_real, L_discriminator

    def evaluate_discriminator_accuracy(self, val_dataloader):
        real_acc = 0
        fake_acc = 0
        self.generator.eval()
        self.discriminator.eval()
        for batch in tqdm(val_dataloader):
            Xs, Xt, _ = self.adapt(batch)

            with torch.no_grad():
                embed = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
                Y_hat = self.generator(Xt, embed, return_attributes=False)                
                fake_D = self.discriminator(Y_hat)
                real_D = self.discriminator(Xs)

            fake_multiscale_acc = 0
            for di in fake_D:
                fake_multiscale_acc += torch.mean((di[0] < 0).float())
            fake_acc += fake_multiscale_acc / len(fake_D)

            real_multiscale_acc = 0
            for di in real_D:
                real_multiscale_acc += torch.mean((di[0] > 0).float())
            real_acc += real_multiscale_acc / len(real_D)
                
        self.generator.train()
        self.discriminator.train()
        
        metrics = {
            'fake_acc': 100 * (fake_acc / len(val_dataloader)).item(),
            'real_acc': 100 * (real_acc / len(val_dataloader)).item()
        }
        return metrics
    
    def evaluate_identification_similarity(self, val_dataloader):
        src_id_sim = 0
        tgt_id_sim = 0
        self.generator.eval()
        for batch in tqdm(val_dataloader):
            Xs, Xt, _ = self.adapt(batch)
            with torch.no_grad():
                src_embed = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
                Y_hat = self.generator(Xt, src_embed, return_attributes=False)

                src_embed = self.mobiface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
                tgt_embed = self.mobiface(F.interpolate(Xt[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
                fake_embed = self.mobiface(F.interpolate(Y_hat[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            
            src_id_sim += (torch.cosine_similarity(src_embed, fake_embed, dim=1)).float().mean()
            tgt_id_sim += (torch.cosine_similarity(tgt_embed, fake_embed, dim=1)).float().mean()

        self.generator.train()

        metrics = {
            'src_similarity': 100 * (src_id_sim / len(val_dataloader)).item(),
            'tgt_similarity': 100 * (tgt_id_sim / len(val_dataloader)).item()
        }
        return metrics

    def generate(self, Xs, Xt, same_person):

        def get_grid_image(X):
            X = X[:8]
            X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0])
            X = (X * 0.5 + 0.5) * 255
            return X

        def make_image(Xs, Xt, Y_hat):
            Xs = get_grid_image(Xs)
            Xt = get_grid_image(Xt)
            Y_hat = get_grid_image(Y_hat)
            return torch.cat((Xs, Xt, Y_hat), dim=1).numpy()

        with torch.no_grad():
            embed = self.arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
            self.generator.eval()
            Y_hat = self.generator(Xt, embed, return_attributes=False)
            self.generator.train()
        
        image = make_image(Xs, Xt, Y_hat)
        if not os.path.exists(f'results/{self.model_dir}'):
            os.makedirs(f'results/{self.model_dir}')
        cv2.imwrite(f'results/{self.model_dir}/{self.iter}.jpg', image.transpose([1,2,0]))


    def get_opt_stats(self, optimizer, type=''):
        stats = {f'{type}_lr' : optimizer.param_groups[0]['lr']}
        return stats

    def adjust_lr(self, optimizer):
        if self.iter <= self.warmup:
            lr = self.lr * self.iter / self.warmup 
        else:
            lr = self.lr * (1 + cos(pi * (self.iter - self.warmup) / (self.max_iters - self.warmup))) / 2
        
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
                checkpoints = glob.glob(f'{path}/discriminator*.pt')
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
                checkpoints = glob.glob(f'{path}/generator*.pt')
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
