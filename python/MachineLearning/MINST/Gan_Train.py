import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import DCGAN
from torch.autograd import Variable


# Hyper Parameters
EPOCH = 100  # 训练的epoch数
Z_DIMENSION = 100  # 生成器的idea数
G_EPOCH = 1  # 判别器的epoch数
NUM_IMG = 100  # 图像的batch size
LR = 0.0003  # 学习率


criterion = nn.BCELoss()
D = DCGAN.Discriminator()
G = DCGAN.Generator(Z_DIMENSION, 1*56*56)
Training_Set, Testing_Set, Training_Loader, Testing_Loader = DCGAN.load_image(NUM_IMG)
D = D.cuda()
G = G.cuda()
d_optimizer = torch.optim.Adam(D.parameters(), lr=LR)
g_optimizer = torch.optim.Adam(G.parameters(), lr=LR)


for count, i in enumerate(range(EPOCH)):
    for (img, label) in Training_Loader:
        img = Variable(img).cuda()
        real_label = Variable(torch.ones(NUM_IMG)).cuda()
        fake_label = Variable(torch.zeros(NUM_IMG)).cuda()

        # Compute Loss Of Real_img
        real_out = D(img)  # output 0 or 1
        d_loss_real = criterion(real_out, real_label)  # loss
        real_scores = real_out

        # Compute Loss Of Fake_img
        z = Variable(torch.randn(NUM_IMG, Z_DIMENSION)).cuda()  # size: (100, 100)
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        # Optimize Discriminator
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        for j in range(G_EPOCH):
            fake_label = Variable(torch.ones(NUM_IMG)).cuda()
            z = Variable(torch.randn(NUM_IMG, Z_DIMENSION)).cuda()  # size: (100, 100)
            fake_img = G(z)
            output = D(fake_img)
            g_loss = criterion(output, fake_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            if (i+1) % 1000 == 0:
                print("[%d/%d] GLoss: %.5f" % (i + 1, G_EPOCH, g_loss.item()))
    print('Epoch [{}/{}], D_loss: {:.6f}, G_loss: {:.6f} '
          'D_real: {:.6f}, D_fake: {:.6f}'.format(i, EPOCH, d_loss.item(), g_loss.item(), real_scores.data.mean(), fake_scores.data.mean()))
    DCGAN.image_show(fake_img, count)
    plt.show()
