# 开发时间 2022/3/17 10:50
import math
import threading
from typing import Sequence

import keras.utils.data_utils
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# from torch import batch_norm
import NetModel
import dataloader
import argparse
from sklearn.model_selection import train_test_split
import NetModel
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.optimizers import SGD,Adam
import matplotlib
import tensorflow as tf
from numpy import array
import matplotlib
# matplotlib.use('module://backend_interagg')

import matplotlib.pyplot as plt
import pickle
import os
import time
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from utils import getGpuMrmory

from keras.models import load_model
from tensorflow.keras.utils import Sequence
#yua
#
def Set_GPU():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #指定第一块GPU可用
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        #设置显存不占满
        tf.config.experimental.set_memory_growth(gpu, True)
        #设置显存占用最大值
        tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
        )


#------------------------------------------------------#
#定义mode
#------------------------------------------------------#
def modelFit(savePath):
    #学习率回调
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    checkpoint = ModelCheckpoint(filepath=savePath, monitor='val_accuracy', mode='auto', save_best_only=True,save_weights_only=False)
    # 模型训练计时
    start = time.time()
    #模型训练
    trainCount=dataloader.getDataDictCount('data/pretrain/'+ninapro+'/trainCount.npy')
    testCount = dataloader.getDataDictCount('data/pretrain/'+ninapro+'/testCount.npy')

    tfrecord_train =dataloader.GetTfFilesList('train',ninapro)    #  'data/pretrain/trainall.tfrecords'
    tfrecord_test = dataloader.GetTfFilesList('test',ninapro)
    dataset_train = dataloader.tfrecords_reader_dataset(tfrecord_train,trainCount,batch_size,ninapro,imuFlag,epoch)
    dataset_test = dataloader.tfrecords_reader_dataset(tfrecord_test,testCount,batch_size,ninapro,imuFlag,epoch)
    # dataset_test = dataloader.tfrecords_reader_dataset(tfrecord_test)
    if model_index == 0:
        model = NetModel.Multi_CNN_Late_ECA(classes, 0.65, ninapro, imuFlag)
    elif model_index == 1:
        model = NetModel.Single_CNN_ECA(classes, 0.5, ninapro, imuFlag)
    model.fit(dataset_train,
              # batch_size=256,
              # validation_data=dataset_test,
              steps_per_epoch=(trainCount// batch_size)+1,
                        epochs=epoch,
              validation_steps=(testCount// batch_size)+1,
              # max_queue_size=30,
             validation_data=dataset_test,

              verbose=2,
              # shuffle=True,
              use_multiprocessing=True,
              # workers=8,
              # validation_data=get_train_batch(valX1, valX2,
              #                                 valX3, valY1, batch_size),
              callbacks=[reduce_lr,checkpoint]
              )
    preds_test = model.evaluate(dataset_test,steps=(testCount// batch_size)+1)
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))
    end = time.time()
    print("模型训练时长:", end - start, "s")

    # 模型保存
    # model.save_weights(savePath, save_format="h5")
    return  model,preds_test[1]

#根据epoch动态改变学习率
def lr_schedule(epoch):
    lr = 1e-1
    if (epoch >= 16)&(epoch<24):
        lr = 1e-2
    elif epoch >= 24:
        lr = 1e-3
    print('Learning rate: ', lr)
    return lr


# predictions = model.predict(x = valX, batch_size=batch_size)
#print(classification_report(valY.argmax(axis=1.h5), predictions.argmax(axis=1.h5), target_names=lb.classes_))
if __name__ == '__main__':
    Set_GPU()
    print("-------------------设置GPU显存按需申请成功----------------------")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--ninapro',default='db7',choices=['db1', 'db5','db7'],
                    help='select ninpro data')
    ap.add_argument('-i', '--imuFlag', action='store_true',
                    help='with imu or no')
    ap.add_argument('-p', '--pretrained', default=True,
                    help='if pretrained give the parser --pretrained, else not')
    ap.add_argument('-m', '--model_index', default='1', choices=['0', '1'],
                    help='if use pretrained model, should give path of pretrained model')
    ap.add_argument('-b', '--batch_size', default=64,
                    help='batch size')
    ap.add_argument('-e', '--epoch', default=28,
                    help='epoch')
    ap.add_argument('-pl', '--plot', default='plot/1.png',
                    help='path to save accuracy/loss plot')
    ap.add_argument('-sm', '--model', default='model/ninapro_db1_multi_001.h5',
                    help='path to save model')
    args = vars(ap.parse_args())
    # ------------------------------------------------------#
    # 参数
    # ------------------------------------------------------#

    subject = args['subject']
    pretrained = args['pretrained']
    batch_size = int(args['batch_size'])
    epoch = int(args['epoch'])
    model_index = int(args['model_index'])
    ninapro=args['ninapro']
    imuFlag = args['imuFlag']
    if ninapro=='db1':
        classes=52
    elif (ninapro=='db5')| (ninapro == 'db7'):
        classes=41
    savePath = 'model_emg+imu/pretrain/'+ninapro +'_'+str(model_index) + '.h5'
    print('dataset:{0},model_index:{1},savePath:{2}'.format(ninapro,model_index,savePath))


    # 2、训练模型 dataPre,input_shape,classes
    model,score= modelFit(savePath)
    print(score)


