import os
import numpy as np

root = '/media/hb/d2221920-26b8-46d4-b6e5-b0eed6c25e6e/lyf毕设/data/classify'
for dataset_name in os.listdir(root):
    base_root = os.path.join(root, dataset_name)
    if not os.path.isdir(base_root): continue
    cls = [x for x in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, x))]
    trainval = list()
    for i, x in enumerate(cls):
        cls_root = os.path.join(base_root, x)
        for pic in os.listdir(cls_root):
            path = os.path.join(cls_root, pic)
            trainval.append((path, i))
    np.random.shuffle(trainval)
    valid = trainval[::5]
    train = [x for x in trainval if x not in valid]
    with open(os.path.join(base_root, 'train.txt'), 'w') as file:
        for x in train:
            file.write(x[0] + '\t' + str(x[1]) + '\n')
    with open(os.path.join(base_root, 'valid.txt'), 'w') as file:
        for x in valid:
            file.write(x[0] + '\t' + str(x[1]) + '\n')





