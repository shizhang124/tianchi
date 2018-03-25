import pandas as pd
import numpy as np
import os

label_count = {'coat_length_labels': 8,
               'collar_design_labels': 5,
               'lapel_design_labels': 5,
               'neck_design_labels': 5,
               'neckline_design_labels': 10,
               'pant_length_labels': 6,
               'skirt_length_labels': 6,
               'sleeve_length_labels': 9
               }

dict_mission = {
    1: 'coat_length_labels',
    2: 'collar_design_labels',
    3: 'lapel_design_labels',
    4: 'neck_design_labels',
    5: 'neckline_design_labels',
    6: 'pant_length_labels',
    7: 'skirt_length_labels',
    8: 'sleeve_length_labels'
}

train_pic_fold = '/media/tang/code/tianchi/data/trainset'
test_pic_fold = '/media/tang/code/tianchi/data/testset_a'
train_label_txt = '/media/tang/code/tianchi/data/trainset/Annotations/label.csv'
test_label_txt = '/media/tang/code/tianchi/data/testset_a/Tests/question.csv'


def load_train_label():
    dict_label = {
        'coat_length_labels': [],
        'collar_design_labels': [],
        'lapel_design_labels': [],
        'neck_design_labels': [],
        'neckline_design_labels': [],
        'pant_length_labels': [],
        'skirt_length_labels': [],
        'sleeve_length_labels': []
    }
    csv_file = pd.read_csv(train_label_txt)
    data_len = csv_file.shape[0]
    print ' '
    # print 'sum train pics:', data_len
    data = np.array(csv_file)
    for line in data:
        label_index = line[2].find('y')
        dict_label[line[1]].append([os.path.join(train_pic_fold, line[0]), label_index])
    # for key in dict_label.keys():
    #     print key,  ' len:', len(dict_label[key])

    return dict_label


def load_test_label():
    dict_label = {
        'coat_length_labels': [],
        'collar_design_labels': [],
        'lapel_design_labels': [],
        'neck_design_labels': [],
        'neckline_design_labels': [],
        'pant_length_labels': [],
        'skirt_length_labels': [],
        'sleeve_length_labels': []
    }
    csv_file = pd.read_csv(test_label_txt)
    data_len = csv_file.shape[0]
    print ' '
    # print 'sum train pics:', data_len
    data = np.array(csv_file)
    for line in data:
        dict_label[line[1]].append([line[0], line[1]])
    # for key in dict_label.keys():
    #     print key,  ' len:', len(dict_label[key])

    return dict_label

# load_train_label()
