import os
import click
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from apex.optimizers import FusedSGD, FusedAdam

from data.faces import LFW
from data.faces import Celebs
from data.transforms import transform_train, transform_val
from trainer import Trainer 
from discriminator_trainer import DiscriminatorTrainer

@click.command()
@click.option('--model_dir', default='models/default')
@click.option('--epochs', default=100)
@click.option('--batch_size', default=128)
@click.option('--eval_every', default=50)
@click.option('--generate_every', default=100)
@click.option('--num_workers', default=4)
def main(model_dir, epochs, batch_size, eval_every, generate_every, num_workers):

    datasets = {'train': Celebs(db_path='datasets/train_db', transform=transform_train),
                'val': LFW(path='datasets/lfw.bin', transform=transform_val)}

    dataloaders = {'train': DataLoader(datasets['train'], 
                                       batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       drop_last=True),

                   'val': DataLoader(datasets['val'],
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)}

    trainer_generator = Trainer(model_dir='mobiface_generator_mse',
                                g_optimizer=FusedAdam,
                                d_optimizer=FusedSGD,
                                num_classes=datasets['train'].num_classes)
    
    # trainer_discriminator = DiscriminatorTrainer(model_dir='checkpoints/mobiface_airface_ce',
    #                                optimizer=FusedSGD,
    #                                lr=lr,
    #                                num_classes=datasets['train'].num_classes)
                
    # trainer_generator.load_discriminator('checkpoints/mobiface_airface_ce', load_last=True)

    for epoch in range(1, epochs + 1):
        trainer_generator.train_loop(dataloaders, eval_every, generate_every)

if __name__ == "__main__":
    main()
