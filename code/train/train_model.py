from data_info import load_train_label, label_count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import os, time
from PIL import Image
from torchvision import transforms
import torchvision.models as models
# from torch.nn import DataParallel
from torch.optim import lr_scheduler
import numpy as np
# import torchvision
import matplotlib.pyplot as plt
import fire
from data_info import load_train_label, label_count, dict_mission


# Load Pic
def default_loader(path):
    return Image.open(path).convert('RGB').resize((330, 330))
    # return Image.open(path).convert('RGB')


def valdata_loader(path):
    return Image.open(path).convert('RGB').resize((299, 299))


class MyDataset(Data.Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for line in label_list:
            if os.path.isfile(line[0]):
                imgs.append(line)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        pic_path, kind = self.imgs[index]
        img = self.loader(pic_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, kind

    def __len__(self):
        return len(self.imgs)


def split_data(data, split_percent):
    length = int(len(data) * split_percent)
    return data[length:], data[:length]


def train(mission_id, BATCH_SIZE, EPOCH, LR, LR_DECAY, DECAY_EPOCH=(30, 50, 70)):
    # Hyper Parameter
    DECAY_EPOCH = [int(x) for x in DECAY_EPOCH]
    # mission_id = 1

    mission_kind = dict_mission[mission_id]
    log_dir = '/media/tang/code/tianchi/logs/' + mission_kind
    model_dir = '/media/tang/code/tianchi/models/' + mission_kind

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    split_percent = 0.1
    IMAGE_CLASS = label_count[mission_kind]

    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_txt = mission_kind + ' googlenet {} BZ_{} EPO_{} LR_{} LR_DE_{} DE_EPO_{}'. \
        format(log_time, BATCH_SIZE, EPOCH, LR, LR_DECAY, DECAY_EPOCH)
    log_path = os.path.join(log_dir, log_txt)
    log_txt = open(log_path, 'wb')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomRotation(5, resample=False, expand=False, center=None),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(4./3., 3./4.)),
        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        # transforms.CenterCrop(224),
        # transforms.TenCrop()
        transforms.ToTensor(),
        normalize,
    ])

    all_data_list = load_train_label()[mission_kind]
    np.random.shuffle(all_data_list)
    train_data_list, val_data_list = split_data(all_data_list, split_percent)

    train_data = MyDataset(label_list=train_data_list, transform=train_transform, loader=default_loader)
    val_data = MyDataset(label_list=val_data_list, transform=val_transform, loader=valdata_loader)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, num_workers=1)
    dataset_sizes = {'train': len(train_data), 'val': len(val_data)}

    print '*********** train: ', mission_id, ':', mission_kind, '*************'
    print mission_kind + ': train pics::', dataset_sizes['train'], 'val pics:', dataset_sizes['val']

    model = models.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, IMAGE_CLASS)
    model.AuxLogits.fc = nn.Linear(768, IMAGE_CLASS)

    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(2048, IMAGE_CLASS)

    # model = torch.load('/media/tang/code/data/models/mixup_models/mixup_cub_googlenet_159_0.7948.pkl')

    # print model
    # for i, param in enumerate(model.parameters()):
    #   param.requires_grad = True
    #   print
    # multi gpu
    # model = DataParallel(model)
    model.cuda()

    # Optimize only the classifier
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=LR_DECAY)
    loss_func = torch.nn.CrossEntropyLoss()

    TOP_ACC = 0.70

    for epoch in range(1, EPOCH + 1):
        scheduler.step()
        lr = scheduler.get_lr()

        # train
        # lr = LR * (LR_DECAY ** (epoch // DECAY_EPOCH))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        info = 'Epoch:{}/{} Lr={:.4f} | '.format(epoch, EPOCH, float(lr[0]))

        model.train()
        time_s = time.time()

        running_loss = 0
        running_corrects = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = Variable(x).cuda(), Variable(y).cuda()
            out1, out2 = model(x)
            loss = loss_func(out1, y) + loss_func(out2, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(out1.data + out2.data, 1)
            running_loss += loss.data[0] * x.size(0)
            running_corrects += torch.sum(preds == y.data)

        use_time = time.time() - time_s
        train_speed = dataset_sizes['train'] // use_time
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = 1.0 * running_corrects / dataset_sizes['train']
        info += '{} Loss:{:.4f} Acc:{:.4f} '.format('train', epoch_loss, epoch_acc)
        info += '[{:.1f}mins/{}pics] | '.format(use_time / 60.0, train_speed)

        # val valset
        model.eval()
        time_s = time.time()
        running_loss = 0
        running_corrects = 0

        for step, (x, y) in enumerate(val_loader):
            # x.ivolatile = True
            x, y = Variable(x).cuda(), Variable(y).cuda()
            out = model(x)
            loss = loss_func(out, y)

            _, preds = torch.max(out.data, 1)
            running_loss += loss.data[0] * x.size(0)
            running_corrects += torch.sum(preds == y.data)

        use_time = time.time() - time_s
        train_speed = dataset_sizes['val'] // use_time
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = 1.0 * running_corrects / dataset_sizes['val']
        info += '{} Loss:{:.4f} Acc:{:.4f} '.format('val', epoch_loss, epoch_acc)
        info += '[{:.1f}mins/{}pics]'.format(use_time / 60.0, train_speed)
        print info
        log_txt.write(info + '\n')
        if TOP_ACC < epoch_acc:
            TOP_ACC = epoch_acc
            model_sava_path = os.path.join(model_dir, mission_kind + '_googlenet_%s_%.4f.pkl' % (epoch, epoch_acc))
            torch.save(model, model_sava_path)
            torch.save(model, os.path.join(model_dir, 'best_model.pkl'))
            acc_info = 'save epoch:{} acc:{}'.format(epoch, epoch_acc)
            print acc_info
            log_txt.write(acc_info + '\n')
    log_txt.close()


if __name__ == '__main__':
    fire.Fire(train)
    # train(1, BATCH_SIZE=46, EPOCH=80, LR=0.01, LR_DECAY=0.1, DECAY_EPOCH='30, 50, 70')
