from turtle import width

import numpy as np
from matplotlib.pyplot import axis, sca
from tensorflow.keras.layers import (Conv2D,Conv1D, Dense, Flatten, Input, MaxPooling2D, Dropout, LocallyConnected2D, BatchNormalization, ReLU, ZeroPadding2D,
                                     Activation,concatenate, GlobalAveragePooling2D,AveragePooling2D, GlobalMaxPooling2D,MaxPooling2D, Lambda, Add, Reshape, multiply, Concatenate,GRU,LSTM,Bidirectional)
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import SGD,Adam
from keras import backend as K
import tensorflow as tf
import gc
import math
import keras
import threading
import os
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import LeakyReLU

dataFormat='channels_last'
useBis=False


def Single_CNN_ECA(classes,dropRate,ninapro):
    inputx0,inputx1,inputx2= GetInput(ninapro)
    print('----input',inputx0)
    # 第一层第一个卷积层
    model = getConv_ECA(inputx0, True,3)
    print('-----------conv1--', model.get_shape().as_list())

    # 第二层卷积层
    model = getConv_ECA(model, False,3)
    print('-----------Conv2--', model.get_shape().as_list())

    # 第三层局部卷积层
    model = getLC(model, True, dropRate)
    print('-----------getLC--', model.get_shape().as_list())

    # 第四层全连接层512 -
    model = getFC(model, True, dropRate)
    print('-----------getFC--', model.get_shape().as_list())
    # 第五层全连接层
    model = getFC(model, False, dropRate)
    print('-----------getFC--', model.get_shape().as_list())
    # 第六层全连接层
    model = Dense(classes, activation='softmax', name='output')(model)
    print('-----------dense--', model.get_shape().as_list())
    model = Model(inputs=[inputx0], outputs=model)
    # 定义优化器
    opt = SGD(learning_rate=0.1, decay=0.0001)
    # 模型编译
    model.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])
    return model


'''卷积层
参数：输入，flag（为True即第一个卷积层需要在前面增加一个batch处理）
'''
def getConv_ECA(input,flag,size):
    if flag:
        batch = BatchNormalization(axis=-1, momentum=0.9)(input)
        batch = eca_block(batch)
        #取消偏置
        conv = Conv2D(filters=64, kernel_size=(size,size), strides=(1, 1), padding='same',use_bias=useBis,data_format=dataFormat)(batch)

    else:
        conv = Conv2D(filters=64, kernel_size=(size,size), strides=(1, 1), padding='same',use_bias=useBis,data_format=dataFormat)(input)
    out = BatchNormalization(axis=-1, momentum=0.9)(conv)
    out = ReLU()(out)
    return out
'''局部卷积层
参数：输入，flag（为True即最后一个卷积层需要在后面增加一个drop处理）
'''
def getLC(input,flag,dropRate):
    lc = LocallyConnected2D(filters=64, kernel_size=1, strides=(1, 1), padding='valid',use_bias=useBis,data_format=dataFormat)(input)
    batch = BatchNormalization(axis=-1, momentum=0.9)(lc)
    rel = ReLU()(batch)
    if flag:
        drop=Dropout(rate=dropRate)(rel)
        return drop
    return rel

'''全连接层
参数：输入，flag（为True即第一个全连接层需要在前面增加一个flatten处理）
'''
def getFC(input,flag,dropRate):
    if flag:
        fla = Flatten()(input)
        print('-----------fla--', fla.get_shape().as_list())
        den=Dense(512,use_bias=useBis,kernel_regularizer=tf.keras.regularizers.l2(0.001))(fla)
    else:
        den = Dense(512,use_bias=useBis,kernel_regularizer=tf.keras.regularizers.l2(0.001))(input)

    den = BatchNormalization(axis=-1, momentum=0.9)(den)
    den = Activation('relu')(den)
    if flag:
        drop = Dropout(rate=dropRate)(den)
        return drop
    else:
        return den

def get3Layers(input,dropRate):
    # 卷积层
    out = getConv_ECA(input, False, 3)
    # 局部连接层
    out = getLC(out, True, dropRate)
    out = getFC(out, True,dropRate)
    return out


def GetInput(ninapro):
    if ninapro == 'db5':
        inputx0 = Input([128,1,64], name='inputx0')
        inputx1 = Input([128,1,42], name='inputx1')
        inputx2 = Input([128,1,48], name='inputx2')
    elif ninapro == 'db1':
        inputx0 = Input([50, 1, 32], name='inputx0')
        inputx1 = Input([50, 1, 22], name='inputx1')
        inputx2 = Input([50, 1, 28], name='inputx2')
    elif (ninapro == 'db2') | (ninapro == 'db3') | (ninapro == 'db7'):
        inputx0 = Input([72, 1, 32], name='inputx0')
        inputx1 = Input([72, 1, 22], name='inputx1')
        inputx2 = Input([72, 1, 28], name='inputx2')
    return inputx0,inputx1,inputx2


def eca_block(input_feature, b=1, gamma=2):
    channel =input_feature.get_shape().as_list()[3]
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    avg_pool = GlobalAveragePooling2D()(input_feature)
    x = Reshape((-1, 1))(avg_pool)
    x = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False, )(x)
    x = Activation('sigmoid')(x)
    x = Reshape((1, 1, -1))(x)
    output = multiply([input_feature, x])
    return output

#

def Multi_CNN_Late_ECA(classes,dropRate,ninapro,imuFlag):
    K.clear_session()
    inputx0, inputx1, inputx2 = GetInput(ninapro)
    #第一个输入
    conv1=getConv_ECA(inputx0,True,3)
    fc1=get3Layers(conv1,dropRate)

    # 第二个输入
    conv2 = getConv_ECA(inputx1, True,3)
    fc2 = get3Layers(conv2,dropRate)

    # 第三个输入
    conv3 = getConv_ECA(inputx2, True,3)
    fc3 = get3Layers(conv3,dropRate)

    if imuFlag:
        inputx3 = Input([36, 1, 20], name='inputx3')
        conv4 = getConv_ECA(inputx3, True,3)
        fc4= get3Layers(conv4, dropRate)
        merge = concatenate([fc1, fc2, fc3, fc4], axis=1)
    else:
        merge= concatenate([fc1, fc2, fc3], axis=1)
    model_late = getFC(merge, False,dropRate)
    #全连接层
    # model_late=merge
    model_late=Dense(classes, activation='softmax',name='output')(model_late)
    if imuFlag:
        model_late = Model(inputs=[inputx0,inputx1,inputx2,inputx3], outputs=model_late)
    else:
        model_late = Model(inputs=[inputx0, inputx1, inputx2], outputs=model_late)

    # 定义优化器
    opt = SGD(learning_rate=0.1, decay=0.0001)
    # 模型编译
    model_late.compile(loss='CategoricalCrossentropy', optimizer=opt, metrics=['accuracy'])
    gc.collect()
    return model_late

if __name__ == '__main__':

    print('11')
