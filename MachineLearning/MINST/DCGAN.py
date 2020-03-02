import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os


def image_show(images, count):
    folder = os.path.exists(r'./GAN_IMAGE')
    if not folder:
        os.makedirs(r'./GAN_IMAGE')
        print('Automatically made a directory!')
    else:
        pass
    images = images.to('cpu')
    images = images.detach().numpy()
    images = images[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]]
    images = 255 * (0.5 * images + 0.5)
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
    print('Processing...')
    plt.tight_layout()
    plt.savefig('./GAN_IMAGE/%d.png' % count, bbox_inches='tight')


def load_image(batch_size, num_of_workers):
    if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        download = True
    else:
        download = False
    Training_Set = datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=download,
    )
    Testing_Set = datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=download,
    )
    Training_Loader = DataLoader(Training_Set, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)
    Testing_Loader = DataLoader(Testing_Set, batch_size=batch_size, shuffle=False, num_workers=num_of_workers)
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
            nn.Linear(1024, 10),
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


def save_model(model1, model2, filename1, filename2):
    folder = os.path.exists(r'./DC-GAN_Networks')
    if not folder:
        os.makedirs(r'./DC-GAN_Networks')
        print('Automatically made a directory!')
    else:
        pass
    torch.save(model1, filename1 + '.pkl')
    torch.save(model2, filename2 + '.pkl')
    state1 = model1.state_dict()
    State1 = state1.copy()
    for key in State1:
        State1[key] = State1[key].clone().cpu()
    torch.save(State1, filename1 + r'_cpu_.pkl')
    state2 = model2.state_dict()
    State2 = state2.copy()
    for key in State2:
        State2[key] = State2[key].clone().cpu()
    torch.save(State2, filename2 + r'_cpu_.pkl')
    print('Saving Successfully!\n')


def show_all(images_all, num_img=9):
    x = images_all[0]
    for i in range(1, len(images_all), 1):
        x=np.concatenate((x, images_all[i]), 0)
    print(x.shape)
    x = 255 * (0.5 * x + 0.5)
    x = x.astype(np.uint8)
    plt.figure(figsize=(9, 10))
    width = x.shape[2]
    gs = gridspec.GridSpec(10, num_img, wspace=0, hspace=0)
    for i, img in enumerate(x):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()


def show(images, num_img=9):
    images = images.detach().numpy()
    images = 255 * (0.5 * images + 0.5)
    images = images.astype(np.uint8)
    plt.figure(figsize=(4, 4))
    width = images.shape[2]
    gs = gridspec.GridSpec(1, num_img, wspace=0, hspace=0)
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(width, width), cmap=plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    plt.tight_layout()
    return width


if __name__ == '__main__':
    G = Generator(110, 1 * 56 * 56)
    D = Discriminator()
    print(G, '\n')
    print(D)
