import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import DCGAN
import numpy as np
from torch.autograd import Variable


# Hyper Parameters
EPOCH = 100  # 训练的epoch数
Z_DIMENSION = 110  # 生成器的idea数,最后十位为label
G_EPOCH = 1  # 判别器的epoch数
NUM_IMG = 100  # 图像的batch size
LR = 0.0003  # 学习率
OPTIMIZER = torch.optim.Adam  # 优化器
CRITERION = nn.BCELoss()  # 损失函数
NUM_OF_WORKERS = 10  # 线程数
N_IDEAS = 100  # 随机数，Z_DEMENSION比它多了tag的数量


D = DCGAN.Discriminator()
G = DCGAN.Generator(Z_DIMENSION, 1*56*56)  #
Training_Set, Testing_Set, Training_Loader, Testing_Loader = DCGAN.load_image(NUM_IMG, NUM_OF_WORKERS)
D = D.cuda()
G = G.cuda()
d_optimizer = OPTIMIZER(D.parameters(), lr=LR)
g_optimizer = OPTIMIZER(G.parameters(), lr=LR)


if __name__ == '__main__':
    for count, i in enumerate(range(EPOCH)):
        for (img, label) in Training_Loader:
            labels_one_hot = np.zeros((NUM_IMG, 10))
            labels_one_hot[np.arange(NUM_IMG), label.numpy()] = 1
            img = Variable(img).cuda()
            real_label = Variable(torch.from_numpy(labels_one_hot).float()).cuda()
            fake_label = Variable(torch.zeros(NUM_IMG, 10)).cuda()

            # Compute Loss Of Real_img
            real_out = D(img)
            d_loss_real = CRITERION(real_out, real_label)
            real_scores = real_out

            # Compute Loss Of Fake_img
            z = Variable(torch.randn(NUM_IMG, Z_DIMENSION)).cuda()
            fake_img = G(z)
            fake_out = D(fake_img)
            d_loss_fake = CRITERION(fake_out, fake_label)
            fake_scores = fake_out

            # Optimize Discriminator
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            for j in range(G_EPOCH):
                fake_label = Variable(torch.ones(NUM_IMG)).cuda()
                z = np.concatenate((torch.randn(NUM_IMG, N_IDEAS).numpy(), labels_one_hot), axis=1)
                z = Variable(torch.from_numpy(z).float()).cuda()
                fake_img = G(z)
                output = D(fake_img)
                g_loss = CRITERION(output, real_label)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        print('Epoch [{}/{}], D_loss: {:.6f}, G_loss: {:.6f} '
              'D_real: {:.6f}, D_fake: {:.6f}'.format(i, EPOCH, d_loss.item(), g_loss.item(), real_scores.data.mean(), fake_scores.data.mean()))
        _, temp_label = torch.max(real_label.to('cpu'), 1)
        print(temp_label.numpy()[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]])
        DCGAN.image_show(fake_img, count)
        plt.show()
    DCGAN.save_model(G, D, r'./DC-GAN/generator', r'./DC-GAN/discriminator')

