import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.applications import inception_resnet_v2
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda
from keras.preprocessing import image
# from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
# import pandas as pd
import numpy as np
import os


def load_data(fold_path, txt_path):
    pics = []
    labels = []
    with open(txt_path, 'rb')as txt:
        for line in txt:
            # print line
            line = line.strip().split()
            img_path = os.path.join(fold_path, line[0])
            label = int(line[1]) - 1

            labels.append(label)
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            # x = preprocess_input(x)
            pics.append(x)
            # x = np.expand_dims(x, axis=0)
            #
    pics = np.array(pics)
    labels = np.array(labels)
    return pics, labels


width = 299
nb_epoch = 100
batch_size = 32

train_txt_path = '/media/tang/code/data/cub/txt/image_class_labels_trainset_jpg.txt'
test_txt_path = '/media/tang/code/data/cub/txt/image_class_labels_testset_jpg.txt'

train_pic_fold = '/media/tang/code/data/cub/img/img_train'
test_pic_fold = '/media/tang/code/data/cub/img/img_test'

X_train, y_train = load_data(train_pic_fold, train_txt_path)
X_test, y_test = load_data(test_pic_fold, test_txt_path)

index = np.arange(len(y_train))
np.random.shuffle(index)
X_train = X_train[index]
y_train = y_train[index]

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', input_shape=(width, width, 3), include_top=False,
                                                   pooling='avg')

input_tensor = Input((width, width, 3))
x = input_tensor
x = Lambda(inception_resnet_v2.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.5)(x)
x = Dense(200, activation='softmax')(x)

model = Model(input_tensor, x)


# model.summary()
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
#                     epochs=nb_epoch, batch_size=batch_size,
#                     shuffle=True, verbose=2)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=nb_epoch, batch_size=batch_size,
                    shuffle=True, verbose=1)
#
# n = len(df)
# y = [np.zeros((n, label_count[x])) for x in label_count.keys()]
