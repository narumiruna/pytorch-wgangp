
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

        for i, (x, _) in enumerate(self.dataloader):
            # train discriminator
            x = x.to(self.device)
            z = torch.randn(x.size(0), 100).to(self.device)

            fake = self.net_g(z).detach()

            loss_d = -self.net_d(x).mean() + self.net_d(fake).mean() + self.c * self.gradient_penalty(x, fake).mean()

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

            loss_d_meter.update(loss_d.item(), number=x.size(0))

            # train generator
            z = torch.randn(x.size(0), 100).to(self.device)

            fake = self.net_g(z)
            loss_g = - self.net_d(fake).mean()

            self.optimizer_g.zero_grad()
            loss_g.backward()
            self.optimizer_g.step()

            loss_g_meter.update(loss_g.item(), number=x.size(0))

        return loss_g_meter.average, loss_d_meter.average

    def save_sample(self, f):
        self.net_g.eval()

        z = torch.randn(64, 100).to(self.device)

        with torch.no_grad():
            fake = self.net_g(z)

        save_image(fake.data, f, normalize=True)

    def gradient_penalty(self, x, fake):
        epsilon = torch.rand(x.size(0), 1, 1, 1).to(self.device)

        interpolates = torch.tensor((epsilon * x + (1 - epsilon) * fake).data, requires_grad=True)
        gradients = autograd.grad(self.net_d(interpolates),
                                  interpolates,
                                  grad_outputs=torch.ones(x.size(0)).to(self.device),
                                  create_graph=True)[0]

        return (gradients.view(x.size(0), -1).norm(2, dim=1) - 1).pow(2)
