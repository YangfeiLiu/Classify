from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
from albumentations import Compose, Flip, RandomResizedCrop, Resize


class ClassifyData(Dataset):
    def __init__(self, size, root, phase):
        if phase == 'train':
            self.data_list = open(os.path.join(root, 'train.txt'), 'r').readlines()
            self.transform = Compose([Flip(),
                                      RandomResizedCrop(width=size, height=size, scale=(0.8, 1.2)),
                                      ])
        else:
            self.data_list = open(os.path.join(root, 'val.txt'), 'r').readlines()
            self.transform = Compose([Resize(width=size, height=size)])

    def __getitem__(self, item):
        data = self.data_list[item].rstrip('\n').split('\t')
        img_path, lab = data[0], int(data[1])
        img = np.array(Image.open(img_path))
        img = self.transform(image=img)['image']
        img = transforms.ToTensor()(img)
        return img, lab

    def __len__(self):
        return len(self.data_list)
