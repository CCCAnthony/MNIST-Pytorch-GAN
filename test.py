import os.path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):  # 生成器
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


# test_data = torch.FloatTensor(128, 100).to(device)
test_data = torch.randn(128, 100).to(device)  # 随机噪声

weight = 'Generator_mnist_1000.pth'
model = Generator(100).to(device)
model.load_state_dict(torch.load(weight))
model.eval()

pred = np.squeeze(model(test_data).detach().cpu().numpy())
# plt.imshow(pred[1].T)
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(((pred[i] + 1) / 2).T,cmap='gray')
    plt.axis('off')

if not os.path.exists('image.png'):
    plt.savefig(fname='image.png', figsize=[20, 20])
else:
    plt.savefig(fname='image_{}.png'.format(weight.split('.')[0]))

plt.show()
