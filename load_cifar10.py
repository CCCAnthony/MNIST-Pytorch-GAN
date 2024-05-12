import torchvision
from torch.utils.data import DataLoader
import os
import numpy as np
import imageio  # 引入imageio包

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 路径将测试集作为验证集
train_path = './dataset/cifar-10-batches-py/train'
test_path = './dataset/cifar-10-batches-py/val'

for i in range(10):
    file_name = train_path + '/'+ str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

for i in range(10):
    file_name = test_path + '/' + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

# 解压 返回解压后的字典
def unpickle(file):
    import pickle as pk
    fo = open(file, 'rb')
    dict = pk.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict


# begin unpickle
root_dir = "./dataset/cifar-10-batches-py"
# 生成训练集图片
print('loading_train_data_')
for j in range(1, 6):
    dataName = root_dir + "/data_batch_" + str(j)  # 读取当前目录下的data_batch1~5文件。
    Xtr = unpickle(dataName)
    print(dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = root_dir + '/train/' + str(Xtr['labels'][i]) + '/' + str(i + (j - 1) * 10000) + '.jpg'
        imageio.imsave(picName, img)  # 使用的imageio的imsave类
    print(dataName + " loaded.")

# 生成测试集图片(将测试集作为验证集)
print('loading_val_data_')
testXtr = unpickle(root_dir + "/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = root_dir + '/val/' + str(testXtr['labels'][i]) + '/' + str(i) + '.jpg'
    imageio.imsave(picName, img)

