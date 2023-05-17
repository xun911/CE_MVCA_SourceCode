# 开发时间 2022/3/17 10:50
from typing import Sequence
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# from torch import batch_norm
import dataloader,dataloader
import argparse
from sklearn.model_selection import train_test_split
import NetModel
from tensorflow.keras.callbacks import LearningRateScheduler
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
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from utils import getGpuMrmory,save_excel
import gc
from statistics import mean
from numpy.random import seed

# seed(666)
from tensorflow.random import set_seed

# set_seed(666)
def Set_GPU():

    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #指定第一块GPU可用
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        #设置显存不占满
        tf.config.experimental.set_memory_growth(gpu,True)
        #设置显存占用最大值
        tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
        )

#------------------------------------------------------#
#定义model
#------------------------------------------------------#
def modelFit(savePath,trainPath,testPath,weightPath):
    # 清空之前model占用的内存，防止OOM
    K.clear_session()
    #dataset
    trainCount=dataloader.countTfRecord(trainPath)
    testCount=dataloader.countTfRecord(testPath)
    if model_index ==3:
        dataset_train = dataloader.tfrecords_reader_dataset_single(trainPath, trainCount, batch_size, ninapro, imuFlag,
                                                                    epoch)
        dataset_test = dataloader.tfrecords_reader_dataset_single(testPath, testCount, batch_size, ninapro, imuFlag,
                                                                   epoch)
    else:

        dataset_train = dataloader.tfrecords_reader_dataset(trainPath, trainCount, batch_size, ninapro, imuFlag, epoch)
        dataset_test = dataloader.tfrecords_reader_dataset(testPath, testCount, batch_size, ninapro, imuFlag, epoch)
    #model
    if model_index==0:
        model = NetModel.Multi_CNN_Late_ECA(classes, 0.65, ninapro, imuFlag)
    elif model_index==1:
        model = NetModel.Single_CNN_ECA(classes,0.5,ninapro,imuFlag)

    # model.summary()
    #学习率回调
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # 模型训练计时
    start = time.time()
    #weights
    model.load_weights(weightPath)
    #模型训练
    #
    H=model.fit(dataset_train,
              # batch_size=256,
              # validation_data=dataset_test,
              steps_per_epoch=(trainCount// batch_size),
              epochs=epoch,
              # max_queue_size=30,
              # validation_data=test_generator,
              verbose=2,
              # shuffle=True,
              # use_multiprocessing=True,
              # workers=8,
              # validation_data=get_train_batch(valX1, valX2,
              #                                 valX3, valY1, batch_size),
              callbacks=[reduce_lr]
              )
    # 绘制训练loss和acc
    # ------------------------------------------------------#
    del dataset_train
    preds_test = model.evaluate(dataset_test,steps=(testCount// batch_size))
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))
    end = time.time()
    fitTime=end - start
    print("模型训练时长:", fitTime, "s")

    # 模型保存
    model.save(savePath, save_format="h5")
    del model

    del dataset_test
    gc.collect()
    # 绘制loss和acc图
    # N = np.arange(0, epoch)
    # plt.style.use('ggplot')
    # plt.figure()
    # plt.plot(N, H.history['loss'], 'g', label='train_loss')
    # plt.plot(N, H.history['val_loss'], 'k', label='val_loss')
    # plt.plot(N, H.history['accuracy'], 'r', label='train_acc')
    # plt.plot(N, H.history['val_accuracy'], 'b', label='val_acc')
    # plt.title("Training Loss and Accuracy (Simple NN)")
    # plt.xlabel("Epoch:#" + str(epoch))
    # plt.ylabel("Loss/Accuracy")
    # plt.legend()
    # plt.show()
    # plt.savefig(args["plot"])

    return  preds_test[1],fitTime

#根据epoch动态改变学习率
def lr_schedule(epoch):
    lr = 1e-1
    if (epoch >= 16)&(epoch<24):
        lr = 1e-2
    elif epoch >= 24:
        lr = 1e-3
    return lr

if __name__ == '__main__':
    # print("-------------------设置GPU显存按需申请成功----------------------")
    Set_GPU()

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--ninapro',  choices=['db1', 'db5','db7'],
                    default='db7',
                 help='select ninpro data')
    ap.add_argument('-i', '--imuFlag', action='store_true',
                    help='with imu or no')
    ap.add_argument('-p', '--pretrained', action='store_true',
                    help='if pretrained give the parser --pretrained, else not')
    ap.add_argument('-m', '--model_index',  default='0',
                    help='Multi_CNN_Late_ECA:0 single_cnn 1')
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
    pretrained = args['pretrained']
    batch_size = int(args['batch_size'])
    epoch = int(args['epoch'])
    model_index=int(args['model_index'])
    ninapro = args['ninapro']
    imuFlag=args['imuFlag']
    if ninapro == 'db1':
        classes = 52
        subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                       '014', '015', '016', '017', '018', '019', '020', '021',
                       '022', '023', '024', '025', '026']
    elif ninapro == 'db5':
        classes = 41
        subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    elif ninapro == 'db7':
        classes = 41
        subjectList = ['000', '001', '002', '003','004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                       '014', '015', '016', '017', '018', '019','021']
    print('model_index--',model_index)
    totalscore=[]

    totalTime=0
    for i in range(len(subjectList)):
        print('第{0}个被试{1}训练开始---------------->'.format(i+1,subjectList[i]))
        savePath = 'model_my/' +ninapro+'/' + subjectList[i] + '.h5'
        trainPath = 'data/pretrain/'+ninapro+'/train_' + subjectList[i] + '.tfrecords'
        testPath = 'data/pretrain/'+ninapro+'/test_' + subjectList[i] + '.tfrecords'
        weightPath='model_my/pretrain/'+ninapro+'_'+str(model_index)+'.h5'
        score,fitTime=modelFit(savePath,trainPath,testPath,weightPath)
        print('被试{0}训练结束，得分为：{1}---------------->'.format(subjectList[i], score))
        gc.collect()
        totalscore.append(score)
        totalTime+=fitTime
    avgscore = mean(totalscore)
    outPath = 'output/' + ninapro + '/ECA.xls'
    save_excel([subjectList, totalscore], ['subject', 'acc'], outPath)
    print('数据集的平均准确率：{0},训练总时长：{1} s'.format(avgscore,totalTime))




