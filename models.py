from torch import nn


class Generator(nn.Module):

    def __init__(self, ch=8):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, ch * 8, 4, 1, 0, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 8, ch * 4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 2, ch, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, 1, 5, 2, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        return self.main(x.view(x.size(0), 100, 1, 1))


class Discriminator(nn.Module):

    def __init__(self, ch=8):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(x.size(0))
