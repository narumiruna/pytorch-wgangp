from torch.utils import data
from torchvision import datasets, transforms


def get_loader(root, batch_size, workers):

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])

    return data.DataLoader(
        datasets.MNIST(root, train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers)
