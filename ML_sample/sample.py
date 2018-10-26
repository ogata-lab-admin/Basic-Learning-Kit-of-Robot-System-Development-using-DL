#!/usr/bin/env python
# coding: utf-8


import os, math
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential

from keras.models import load_model

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
#
#from keras.callbacks import TensorBoard
#from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.utils import plot_model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import h5py
import pickle

#import tensorflow as tf

#gpuConfig = tf.ConfigProto(
#    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95),
#    device_count={'GPU': 0})
#sess = tf.Session(config=gpuConfig)


# フォルダを指定
log_dir = '~'

# csvファイルの読み込み
c = pd.read_csv(os.path.join(log_dir, 'joints.csv'))
Y = [y for y in zip((c['x']-0.12)/0.12, (c[' y']+0.12)/0.24, (c[' theta']+math.pi)/(2*math.pi))]
X = [img_to_array(load_img(os.path.join(log_dir, png.strip()), target_size=(128,128)))/256 for png in c[' ImageFilename']]

# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1234)


# モデルの生成
# input(64,64,3)->conv2D(5,5)->conv2D(5,5)->Dropout(0.25)->conv2D(3,3)->conv2D(3,3)
#  ->MaxPooling2D(2,2)->Dropout(0.25)->fc(512)->fc(256)->fc(3)
model = Sequential()


model.add(Conv2D(64, (5, 5), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Dense(64))
#model.add(Activation('relu'))
#odel.add(Dropout(0.5))

model.add(Dense(3))       # クラスは2個
model.add(Activation('linear'))



#model.load_weights("./param.hdf5", by_name=True)

# コンパイル
model.compile(loss='mean_squared_error',
              optimizer='SGD',
              metrics=['accuracy'])

model.save('my_model.h5')

#tbcb = TensorBoard(log_dir='./graph',
#                   histogram_freq=0, write_graph=True)

checkpoint = ModelCheckpoint(
    filepath = os.path.join(
        'weight_{epoch:02d}.hdf5'),
        save_weights_only=True,
        period=500)
#    save_best_only=True)

# 実行,出力はなしで設定(verbose=0)
history = model.fit(X_train, y_train, batch_size=10, epochs=3000,
                   validation_data = (X_test, y_test),callbacks=[checkpoint]), verbose = 0)

#
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# パラメータの保存
model.save_weights('param.hdf5')

json_string = model.to_json()
open('model_log.json', 'w').write(json_string)

# モデルの可視化
model_json = model.to_json()
with open("model.json", mode = 'w') as f:
    f.write(model_json)

model = None
with open('model.json') as f:
    model= model_from_json(f.read())
plot_model(model, to_file='model.png', show_shapes=True)

# グラフの作成
with open("history.pickle", mode='wb') as f:
    pickle.dump(history.history, f)

history = None
with open('history.pickle', mode='rb') as f:
    history = pickle.load(f)

plt.plot(history['loss'], label="loss",)
plt.plot(history['val_loss'], label="val_loss")
plt.plot(history['acc'], label="acc",)
plt.plot(history['val_acc'], label="val_acc")
plt.title('train history')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='center right')
plt.show()
