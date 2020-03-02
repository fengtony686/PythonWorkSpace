import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn


def image_show(images, count):
    images = images.to('cpu')
    images = images.detach().numpy()
    images = images[[5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95]]
    images = 255*(0.5*images+0.5)
    images = images.astype(np.uint8)
    grid_length = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4, 4))
    width = images.shape[2]
    gs = gridspec.GridSpec(grid_length, grid_length, wspace=0, hspace=0)
    print(images.shape)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    print('Processing')
    plt.tight_layout()
    plt.savefig('./GAN_IMAGE/%d.png' % count, bbox_inches='tight')


def load_image(batch_size):
    Training_Set = datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=False,
    )
    Testing_Set = datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=False,
    )
    Training_Loader = DataLoader(Training_Set, batch_size=batch_size, shuffle=True, num_workers=10)
    Testing_Loader = DataLoader(Testing_Set, batch_size=batch_size, shuffle=False, num_workers=10)
    return Training_Set, Testing_Set, Training_Loader, Testing_Loader


# Discriminator Of DC-GAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Generator Of DC-GAN
class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.BatchNorm = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.Gen = nn.Sequential(
            nn.Conv2d(1, 50, 3, 1, 1),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50, 25, 3, 1, 1),
            nn.BatchNorm2d(25),
            nn.ReLU(True),
            nn.Conv2d(25, 1, 2, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.BatchNorm(x)
        x = self.Gen(x)
        return x


if __name__ == '__main__':
    G = Generator(100, 1*56*56)
    D = Discriminator()
    print(G, '\n')
    print(D)
