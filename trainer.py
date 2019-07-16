import os

import torch
from torch import autograd, optim
from torchvision.utils import save_image

from models import Discriminator, Generator
from utils import AverageMeter


class Trainer(object):

    def __init__(self, net_g, net_d, optimizer_g, optimizer_d, dataloader, device, c=10.0):
        self.net_g = net_g
        self.net_d = net_d
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.dataloader = dataloader
        self.device = device
        self.c = c

    def train(self):
        self.net_g.train()

        loss_g_meter = AverageMeter()
        loss_d_meter = AverageMeter()

        for real, _ in self.dataloader:
            # train discriminator
            real = real.to(self.device)
            z = torch.randn(real.size(0), 100).to(self.device)

            fake = self.net_g(z).detach()

            loss_d = -self.net_d(real).mean()
            loss_d += self.net_d(fake).mean()
            loss_d += self.c * self.gradient_penalty(real, fake).mean()

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

            loss_d_meter.update(loss_d.item(), number=real.size(0))

            # train generator
            z = torch.randn(real.size(0), 100).to(self.device)

            fake = self.net_g(z)
            loss_g = -self.net_d(fake).mean()

            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            loss_g_meter.update(loss_g.item(), number=real.size(0))

        return loss_g_meter.average, loss_d_meter.average

    def save_sample(self, filename):
        self.net_g.eval()

        z = torch.randn(64, 100).to(self.device)

        with torch.no_grad():
            fake = self.net_g(z)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_image(fake.data, filename, normalize=True)

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)

        interpolates = epsilon * real + (1 - epsilon) * fake
        interpolates = interpolates.clone().detach().requires_grad_(True)
        gradients = autograd.grad(self.net_d(interpolates),
                                  interpolates,
                                  grad_outputs=torch.ones(batch_size, device=self.device),
                                  create_graph=True)[0]

        return (gradients.view(batch_size, -1).norm(2, dim=1) - 1).pow(2)
