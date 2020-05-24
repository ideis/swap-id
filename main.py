import click

from torch.utils.data import DataLoader
from torch.optim import Adam

from data.faces import FFHQ
from data.transforms import transform_train, transform_val
from trainer import Trainer

@click.command()
@click.option('--model_dir', default='models/default')
@click.option('--epochs', default=20)
@click.option('--train_batch_size', default=4)
@click.option('--val_batch_size', default=10)
@click.option('--lr', default=4e-4)
@click.option('--warmup', default=1000)
@click.option('--max_iters', default=100000)
@click.option('--eval_every', default=250)
@click.option('--generate_every', default=500)
@click.option('--save_every', default=5000)
@click.option('--num_workers', default=16)
def main(model_dir, epochs, train_batch_size, val_batch_size, lr, warmup, max_iters, eval_every, generate_every, save_every, num_workers):

    datasets = {"train": FFHQ("datasets/ffhq", transform_train, same_person_prob=0.5),
                "val": FFHQ("datasets/ffhq_val", transform_val, same_person_prob=0)}

    dataloaders = {'train': DataLoader(datasets['train'], 
                                       batch_size=train_batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       drop_last=True),
                   'val': DataLoader(datasets['val'], 
                                       batch_size=val_batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=True)}

    trainer = Trainer(model_dir='faceshifter',
                      g_optimizer=Adam,
                      d_optimizer=Adam,
                      lr=lr,
                      warmup=warmup,
                      max_iters=max_iters)
    
    trainer.load_discriminator('checkpoints/faceshifter', load_last=True)
    trainer.load_generator('checkpoints/faceshifter', load_last=True)

    for epoch in range(1, epochs + 1):
        trainer.train_loop(dataloaders, eval_every, generate_every, save_every)

if __name__ == "__main__":
    main()
