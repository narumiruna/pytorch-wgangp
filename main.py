import argparse

import torch

from models import Discriminator, Generator
from trainer import Trainer

from loader import get_loader
from torch import optim

import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # networks
    net_g = Generator().to(device)
    net_d = Discriminator().to(device)

    print(net_g)
    print(net_d)

    # optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=config.lr, betas=(0.5, 0.999))

    print(optimizer_d)
    print(optimizer_g)
    # data loader
    dataloader = get_loader(config.root, config.batch_size, config.workers)

    trainer = Trainer(net_g, net_d, optimizer_g, optimizer_d, dataloader, device)

    os.makedirs('results', exist_ok=True)

    for epoch in range(config.epochs):
        loss_g, loss_d = trainer.train()

        print('Train epoch: {}/{},'.format(epoch + 1, config.epochs),
              'loss g: {:.6f}, loss d: {:.6f}.'.format(loss_g, loss_d))

        trainer.save_sample('results/sample_{:02d}.jpg'.format(epoch + 1))


if __name__ == '__main__':
    main()
