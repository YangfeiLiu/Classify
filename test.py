from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
import numpy as np
import json
from models.resnext import resnext50
from tqdm import tqdm


with open('./cls.json', 'r') as f:
    cls2label = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ClassifyData(Dataset):
    def __init__(self, root):
        self.data_list = list()
        for cls in os.listdir(root):
            label = cls2label[cls]
            path1 = os.path.join(root, cls)
            for name in os.listdir(path1):
                path2 = os.path.join(path1, name)
                self.data_list.append([path2, label])

    def normal(self, img):
        return img / 255.

    def __getitem__(self, item):
        datas = self.data_list[item]
        img_path, lab = datas[0], int(datas[1])
        img = np.array(Image.open(img_path))
        img = self.normal(img)
        img = torch.from_numpy(img).permute((2, 0, 1)).float()
        return img, lab

    def __len__(self):
        return len(self.data_list)


def infer():
    model_path = './resnext50_NWPU-RESISC45.pth'
    batch_size = 256
    net = resnext50(num_classes=45).to(device)
    pretrain = torch.load(model_path)
    net.load_state_dict(pretrain)

    test_data = ClassifyData(root='./test_data')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16)
    tbar = tqdm(test_loader)
    test_accu = 0
    net.eval()
    with torch.no_grad():
        for i, (img, lab) in enumerate(tbar):
            tbar.set_description("Testing->>")
            img = img.to(device)
            b = img.size(0)
            prob = net.forward(img)
            pred = prob.data.max(1)[1].cpu()
            accu = float(pred.eq(lab.data).sum()) / b
            test_accu = ((test_accu * i) + accu) / (i + 1)
    print("test accuracy=%.6f" % test_accu)


if __name__ == '__main__':
    infer()