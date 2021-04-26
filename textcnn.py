# -*- coding: utf-8 -*-
# @Time : 2020/11/4
# @Author :
# @File : textcnn.py
# @Software: PyCharm
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# ===================== set random  ===========================

import numpy as np
import random as rn
np.random.seed(0)
rn.seed(0)
#tf.random.set_seed(0)
# =============================================================

# +
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
# -

class TextCNN(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=[1,2,3],
                 kernel_regularizer=None,
                 last_activation='softmax'
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [1,2,3]
        :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
        :param class_num:
        :param last_activation:
        '''
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.conv1s = []
        self.avgpools = []
        for kernel_size in kernel_sizes:
            self.conv1s.append(Conv1D(filters=128, kernel_size=kernel_size, activation='relu', kernel_regularizer=kernel_regularizer))
            self.avgpools.append(GlobalMaxPooling1D())
        self.classifier = Dense(class_num, activation=last_activation, )

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](emb) # (batch_size, maxlen-kernel_size+1, filters)
            c = self.avgpools[i](c) # # (batch_size, filters)
            conv1s.append(c)
        x = Concatenate()(conv1s) # (batch_size, len(self.kernel_sizes)*filters)
        output = self.classifier(x)
        return output

    def build_graph(self, input_shape):
        '''自定义函数，在调用model.summary()之前调用
        '''
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)
def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        print('Bulid Model...')
        self.create_model()

    def create_model(self):
        model = TextCNN(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                         kernel_sizes=[2,3,5],
                         kernel_regularizer=None,
                         last_activation='softmax')
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        model.build_graph(input_shape=(None, maxlen))
        model.summary()
        self.model =  model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\TextCNN-epoch-5', checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []
        if use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='loss', patience=7, mode='max')
            callback_list.append(early_stopping)
        if checkpoint_path is not None:
            # save model
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                             monitor='val_accuracy',
                                             mode='max',
                                             save_best_only=True,
                                             save_weights_only=True,
                                             verbose=1,
                                             period=2,
                                             )
            callback_list.append(cp_callback)
        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/TextCNN-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=2,
                  callbacks=self.callback_list,
                  validation_data=(x_val, y_val))

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)

# ================  params =========================
class_num = 2
maxlen = 400
embedding_dims = 200
epochs = 10
batch_size = 32
max_features = 5000

MODEL_NAME = 'company_manager_relation_TextCNN'

use_early_stop=True
tensorboard_log_dir = './logs/{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = './save_model_dir/'+MODEL_NAME+'/cp-{epoch:04d}.ckpt'
#  ====================================================================

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print('Pad sequences (samples x time)...')
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size
                           )
model_hepler.get_callback(use_early_stop=use_early_stop, checkpoint_path=checkpoint_path)
model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)
print('Test...')
result = model_hepler.model.predict(x_test)
test_score = model_hepler.model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print("test loss:", test_score[0], "test accuracy", test_score[1])



model_hepler = ModelHepler(class_num=class_num,
                           maxlen=maxlen,
                           max_features=max_features,
                           embedding_dims=embedding_dims,
                           epochs=epochs,
                           batch_size=batch_size
                           )
model_hepler.load_model(checkpoint_path=checkpoint_path)
# 重新评估模型
loss, acc = model_hepler.model.evaluate(x_test, y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
