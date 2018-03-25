import keras
from keras.applications import inception_resnet_v2
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import pandas as pd
import numpy as np
import os
label_count = {'coat_length': 8,
               'collar_design': 5,
               'lapel_design': 5,
               'neck_design': 5,
               'neckline_design': 10,
               'pant_length': 6,
               'skirt_length': 6,
               'sleeve_length': 9
               }

data_list = pd.read_csv('/media/tang/code/tianchi/data/trainset/Annotations/label.csv')
print data_list.shape
print type(data_list)


def load_data(fold_path, txt_path):
    data_list = pd.read_csv(txt_path)
    print data_list[0]
    pic_paths = []
    labels_kinds = []
    labels = []

    for line in data_list:
        print line
        img_path = os.path.join(fold_path, line[0])
        label_kind  = line[1]
        label = line[2].find('y')
        pic_paths.append(img_path)
        labels_kinds.append(label_kind)
        labels.append(label)

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return


x_train, y_train = load_data('/media/tang/code/tianchi/data/trainset',
                             '/media/tang/code/tianchi/data/trainset/Annotations/label.csv')

width = 299

base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', input_shape=(width, width, 3), include_top=False,
                                                   pooling='avg')

input_tensor = Input((width, width, 3))
x = input_tensor
x = Lambda(inception_resnet_v2.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.5)(x)
x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]

model = Model(input_tensor, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit_generator(0)

n = len(df)
y = [np.zeros((n, label_count[x])) for x in label_count.keys()]
