import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from models import densenet, googlenet, resnet, resnext, senet, shufflenet, vgg, xception
from data import ClassifyData
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import os
from loguru import logger
from tensorboardX import SummaryWriter


parse = argparse.ArgumentParser()
parse.add_argument('--epoch', default=200)
parse.add_argument('--lr', default=0.0001)
parse.add_argument('--train_bs', default=64)
parse.add_argument('--size', default=256)
parse.add_argument('--test_bs', default=64)
parse.add_argument('--num_workers', default=16)
parse.add_argument('--root', default='')
parse.add_argument('--model_path', default='')
parse.add_argument('--best_accu', default=0)
args = parse.parse_args()

dataset = [x for x in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, x))]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def adjust_learning_rate(optimizer, epoch, lr):
    wd = 1e-4
    milestone = 20  # after epoch milestone, lr is reduced exponentially
    if epoch > milestone:
        lr = lr * (0.98 ** (epoch - milestone))
        wd = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd


def train():
    net = model.to(device)

    optimizer = Adam(net.parameters(), lr=args.lr)

    train_data = ClassifyData(size=args.size, root=os.path.join(args.root, ds), phase='train')
    train_loader = DataLoader(train_data, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)

    val_data = ClassifyData(size=args.size, root=os.path.join(args.root, ds), phase='val')
    val_loader = DataLoader(val_data, batch_size=args.test_bs, shuffle=True, num_workers=args.num_workers)

    cnt = 0
    for epoch in range(1, args.epoch+1):
        lr = optimizer.param_groups[0]['lr']
        adjust_learning_rate(optimizer, epoch, lr)
        logger.info("epoch:%d\t lr:%.8f" % (epoch, lr))
        train_loss = 0
        train_accu = 0
        net.train()
        tbar = tqdm(train_loader)
        for i, (img, lab) in enumerate(tbar):
            tbar.set_description("train_accu=%.6f" % train_accu)
            tbar.set_postfix({"train_loss": train_loss})
            b = img.size(0)
            img = img.to(device)
            lab = lab.to(device)
            optimizer.zero_grad()

            out = net.forward(img)
            loss = F.cross_entropy(out, lab)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss = (train_loss * i + float(loss)) / (i + 1)
                pred = out.data.max(1)[1]
                accu = float(pred.eq(lab.data).sum()) / b
                train_accu = ((train_accu * i) + accu) / (i + 1)
        logger.info("train_loss=%.6f \t train_accu=%.6f" % (train_loss, train_accu))
        writer.add_scalar("train/train_accuracy", train_accu, epoch)
        writer.add_scalar("train/train_loss", train_loss, epoch)

        net.eval()
        with torch.no_grad():
            val_accu = 0
            val_loss = 0
            val_tbar = tqdm(val_loader)
            for i, (img, lab) in enumerate(val_tbar):
                val_tbar.set_description("val_accu=%.6f" % val_accu)
                b = img.size(0)
                img = img.to(device)
                lab = lab.to(device)

                out = net.forward(img)
                loss = F.cross_entropy(out, lab)

                val_loss = (val_loss * i + float(loss)) / (i + 1)
                pred = out.data.max(1)[1]
                accu = float(pred.eq(lab.data).sum()) / b
                val_accu = ((val_accu * i) + accu) / (i + 1)
            writer.add_scalar("valid/valid_accuracy", val_accu, epoch)
            writer.add_scalar("valid/valid_loss", val_loss, epoch)
        logger.info("val_loss=%.6f \t val_accu=%.6f" % (val_loss, val_accu))
        if val_accu > args.best_accu:
            args.best_accu = val_accu
            cnt = 0
            try:
                torch.save(net.state_dict(), os.path.join(args.model_path, '%s_%s.pth' % (net_name, ds)),
                           _use_new_zipfile_serialization=False)
            except:
                torch.save(net.state_dict(), os.path.join(args.model_path, '%s_%s.pth' % (net_name, ds)))
            logger.info("%d saved" % epoch)
        else:
            cnt += 1
            if cnt == 20:
                logger.info("early stop")
                break


for ds in dataset:
    data_path = os.path.join(args.root, ds)
    cls = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]
    num_class = len(cls)
    models = {"googlenet": googlenet.googlenet(num_class), "vgg16": vgg.vgg16_bn(num_class), "vgg19": vgg.vgg19_bn(num_class),
              "densenet121": densenet.densenet121(num_class), "densenet161": densenet.densenet161(num_class),
              "resnet34": resnet.resnet34(num_class), "resnet50": resnet.resnet50(num_class),
              "resnet101": resnet.resnet101(num_class), "seresnet34": senet.seresnet34(num_class),
              "seresnet50": senet.seresnet50(num_class), "seresnet101": senet.seresnet101(num_class),
              "resnext34": resnext.resnext34(num_class), "resnext50": resnext.resnext50(num_class),
              "resnext101": resnext.resnext101(num_class),
              "shufflenet": shufflenet.shufflenet(num_class), "xception": xception.xception(num_class)}
    for net_name in models.keys():
        writer = SummaryWriter('./runs/%s_%s/' % (ds, net_name))
        model = models[net_name]
        logger.add('./log/%s_%s_{time}.log' % (ds, net_name), level="INFO")
        logger.info("net:%s\t dataset:%s\t num_class:%d" % (net_name, ds, num_class))
        train()
        writer.close()
