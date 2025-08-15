from torchvision import datasets
from tqdm import tqdm
import os


train_data = datasets.MNIST(root="./data/", train=True, download=True)
test_data = datasets.MNIST(root="./data/", train=False, download=True)
saveDirTrain = './train_images'
saveDirTest = './test_images'

if not os.path.exists(saveDirTrain):
    os.mkdir(saveDirTrain)
if not os.path.exists(saveDirTest):
    os.mkdir(saveDirTest)


def save_img(data, save_path, num):
    for i in tqdm(range(len(data))):
        if i == num:
            break
        img, label = data[i]
        img.save(os.path.join(save_path, str(i) + '-' + str(label) + '.png'))


save_img(train_data, saveDirTrain,10)
save_img(test_data, saveDirTest,10)
