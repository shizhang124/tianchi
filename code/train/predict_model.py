import torch
from torch.autograd import Variable
import torch.utils.data as Data
import os, time
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
# from torch.nn import DataParallel
import numpy as np
import fire
import pandas as pd
from data_info import load_test_label, label_count, test_pic_fold, dict_mission


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
            pic_path = os.path.join(test_pic_fold, line[0])
            if os.path.isfile(pic_path):
                imgs.append(line)
            else:
                print 'test pic not exist:', pic_path
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        tmp_pic_path, mission_kind = self.imgs[index]
        pic_path = os.path.join(test_pic_fold, tmp_pic_path)
        img = self.loader(pic_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, tmp_pic_path, mission_kind

    def __len__(self):
        return len(self.imgs)


def split_data(data, split_percent):
    length = int(len(data) * split_percent)
    return data[length:], data[:length]


def test(mission_id, BATCH_SIZE=64):
    # mission_id = 1
    mission_kind = dict_mission[int(mission_id)]
    test_model_path = os.path.join('/media/tang/code/tianchi/models', mission_kind, 'best_model.pkl')
    predict_csv_path = os.path.join('/media/tang/code/tianchi/predicts', mission_kind + '.csv')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomRotation(5, resample=False, expand=False, center=None),
    #     # transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(4./3., 3./4.)),
    #     # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    #     transforms.RandomCrop(299),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    val_transform = transforms.Compose([
        # transforms.CenterCrop(224),
        # transforms.TenCrop()
        transforms.ToTensor(),
        normalize,
    ])

    test_data_list = load_test_label()[mission_kind]
    val_data = MyDataset(label_list=test_data_list, transform=val_transform, loader=valdata_loader)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, num_workers=1)
    dataset_sizes = {'val': len(val_data)}

    print '********test*************'
    print mission_id, ' ', mission_kind + ':test pics:', dataset_sizes['val']

    model = torch.load(test_model_path)
    model.cuda()
    # loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 2):
        # val valset
        model.eval()
        time_s = time.time()
        running_loss = 0
        running_corrects = 0

        predict_results = []
        for step, (x, tmp_img_path, tmp_mission_kind) in enumerate(val_loader):
            x = Variable(x).cuda()
            out = model(x)
            # loss = loss_func(out, y)
            out = F.softmax(out)

            _, preds = torch.max(out.data, 1)
            probs = out.cpu().data.numpy()
            # running_loss += loss.data[0] * x.size(0)
            # running_corrects += torch.sum(preds == y.data)

            # write info
            for i in range(len(tmp_img_path)):
                tmp_label = '{:.4f}'.format(probs[i][0])
                for j in range(1, label_count[mission_kind]):
                    tmp_label += ";" + '{:.4f}'.format(probs[i][j])
                    # if j == preds[i]:
                    #     tmp_label += 'y'
                    # else:
                    #     tmp_label += 'n'

                info = [tmp_img_path[i], tmp_mission_kind[i], tmp_label]
                predict_results.append(info)

        use_time = time.time() - time_s
        train_speed = dataset_sizes['val'] // use_time
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = 1.0 * running_corrects / dataset_sizes['val']
        info = '{} Loss:{:.4f} Acc:{:.4f} '.format('val', epoch_loss, epoch_acc)
        info += '[{:.1}mins/{}pics]'.format(use_time / 60.0, train_speed)
        print info

        # write csv
        df = pd.DataFrame(predict_results)
        df.to_csv(predict_csv_path, sep=',', header=False, index=False)
        # log_txt.write(info + '\n')
        # log_txt.close()


if __name__ == '__main__':
    fire.Fire(test)
    # test(1, BATCH_SIZE=46, EPOCH=60, LR=0.01, LR_DECAY=0.1, DECAY_EPOCH=25)
