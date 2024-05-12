import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

class MyDataset(torch.utils.data.Dataset):  # 载入自己的数据集合
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.names_list = []

        for dirs in os.listdir(self.root_dir):
            dir_path = self.root_dir + '/' + dirs
            for imgs in os.listdir(dir_path):
                img_path = dir_path + '/' + imgs
                self.names_list.append((img_path, dirs))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, index):
        image_path, label = self.names_list[index]
        if not os.path.isfile(image_path):
            print(image_path + '不存在该路径')
            return None
        image = Image.open(image_path)

        # label = np.array(label).astype(int)
        # label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)

        return image


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


class Discriminator(nn.Module):  # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity


def gen_img_plot(model, test_input):  # 测试生成器
    pred = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((pred[i] + 1) / 2)
        plt.axis('off')
    plt.show(block=False)
    plt.savefig(fname="mnist.png")
    plt.pause(3)  # 停留0.5s
    plt.close()


def D_loss_and_G_loss_situation(D_loos, G_loss, epochs):
    assert len(D_loos) == len(G_loss) == epochs, "congirm the coordinate length"
    X_array = np.arange(epochs)
    Y_g_loss = G_loss
    Y_d_loss = D_loos
    plt.plot(X_array, Y_g_loss, label="G_loss", color='blue')
    plt.plot(X_array, Y_d_loss, linestyle="--", label="D_loss", color='red')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("G_loss & D_loss")
    plt.legend()
    plt.savefig(fname="loss_mnist.png")
    plt.show()


# 调用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 超参数设置
lr = 0.0001
batch_size = 128
latent_dim = 100  # 潜在空间维度 要小于图片尺寸（自编码器）
epochs = 100

# latent_dim 是潜在空间（latent space）的维度。在生成对抗网络（GAN）中，潜在空间是一个低维度的向量空间，用来表示生成器网络的输入。
# 生成器网络将这个低维的潜在向量转换为高维的输出，比如图像。latent_dim 决定了潜在空间的维度大小。
# 定义 latent_dim 的目的是为了控制生成器网络的输入维度。
# 通过调整 latent_dim 的大小，我们可以改变生成器网络对于同一潜在向量的输出结果。
# 较小的 latent_dim 可能会导致生成器网络无法捕捉到数据集中的更多细节，而较大的 latent_dim 可能会导致过拟合或者过多噪声。
# latent_dim 的作用是提供一个潜在空间，使得我们可以通过在这个空间中进行采样来生成不同的输出，从而实现对生成器网络输出的控制和变化。

# 数据集载入和数据变换
# 训练数据
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False) # 下载压缩包的数据集
train_dataset = MyDataset('./images_mnist/train', transform=transform)  # images中的图片数据集合
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 测试数据 torch.randn()函数的作用是生成一组均值为0，方差为1(即标准正态分布)的随机数
# test_data = torch.randn(batch_size, latent_dim).to(device)
test_data = torch.FloatTensor(batch_size, latent_dim).to(device)

# 实例化生成器和判别器，并定义损失函数和优化器
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 记录loss绘图
G_loss = []
D_loss = []

# 开始训练模型
for epoch in range(epochs):
    for i, (imgs) in enumerate(train_loader):  # 60000/128为一个epoch
        batch_size = imgs.shape[0]
        real_imgs = imgs.to(device)

        # 训练判别器
        z = torch.FloatTensor(batch_size, latent_dim).to(device)
        z.data.normal_(0, 1)
        fake_imgs = generator(z)  # 生成器生成假的图片

        real_labels = torch.full((batch_size, 1), 1.0).to(device)
        fake_labels = torch.full((batch_size, 1), 0.0).to(device)

        real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        z.data.normal_(0, 1)
        fake_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(fake_imgs), real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        print(f"Epoch [{epoch + 1}/{epochs}] Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), "Generator_mnist.pth")

    G_loss.append(int(g_loss.item()))
    D_loss.append(int(d_loss.item()))
    # print(f"Epoch [{epoch + 1}/{epochs}] Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}")

# 测试模型
gen_img_plot(generator, test_data)
# loss值绘图
D_loss_and_G_loss_situation(G_loss, D_loss, epochs)
