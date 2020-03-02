import torch
import DCGAN
import numpy as np
import matplotlib.pyplot as plt


NUM_IMG = 9
Z_DEMENSION = 110
N_IDEAS = 100


D = DCGAN.Discriminator()
G = DCGAN.Generator(Z_DEMENSION, 3136)
D.load_state_dict(torch.load(r'./DC-GAN-Networks/discriminator_cpu_.pkl'))
G.load_state_dict(torch.load(r'./DC-GAN-Networks/generator_cpu_.pkl'))


lis = []
for i in range(10):
    z = torch.randn((NUM_IMG, N_IDEAS))
    x = np.zeros((NUM_IMG, Z_DEMENSION-N_IDEAS))
    x[:, i] = 1
    z = np.concatenate((z.numpy(), x), 1)
    z = torch.from_numpy(z).float()
    fake_img = G(z)
    lis.append(fake_img.detach().numpy())
    output = D(fake_img)
    DCGAN.show(fake_img)
    plt.savefig('./GAN_IMAGE/Test %d.png' % i, bbox_inches='tight')

DCGAN.show_all(lis)
plt.savefig('./GAN_IMAGE/Test_All.png', bbox_inches='tight')
plt.show()
