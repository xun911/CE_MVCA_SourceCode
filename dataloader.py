import gc

import numpy as np
import os
# from imutils import paths
from tqdm import tqdm
import random
import mat4py
import platform
import scipy.io as scio
from scipy.signal import butter, lfilter, filtfilt
import tensorflow as tf
from utils import get_sig_img,genIndex,downsample,get_segments_image,min_max,butter_filter
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,scale
import warnings
from scipy.stats import stats




def INiaPro_feature_split_single(datapath,subject,feature,cflag,ninapro):

  if ninapro=='db5':
    # 训练集 第1、3,4,6,8,9,10
    trainindex = ['000', '002', '003', '005']
    testindex = ['001', '004']
    classes=41
    channel=16
    ge_first_index=0
  elif ninapro=='db1':
    trainindex = ['000', '002', '003', '005', '007', '008', '009']
    testindex = ['001', '004', '006']
    classes = 52
    channel = 10
    ge_first_index = 1
  elif ninapro=='db7':
    # 训练集 第1、3,4,6,8,9,10
    trainindex = ['000', '002', '003', '005']
    testindex = ['001', '004']
    classes=41
    channel=12
    ge_first_index=0
  trainData = []
  testData = []
  trainLabel = []
  testLabel = []
  trainPaths = []
  testPaths = []
  datapath=datapath+'/'+subject
  sigmig_index=genIndex(channel)

  #生成训练集和测试集路径
  for i in tqdm(range(classes)):
    geindex=str(i+ge_first_index).rjust(3, '0')
    path1=datapath+'/'+geindex
    for j in range(len(trainindex)):
      filePath=path1+'/'+subject+'_'+geindex+'_'+trainindex[j]+'_'+feature+'.mat'
      trainPaths.append(filePath)
    for k in range(len(testindex)):
      filePath=path1+'/'+subject+'_'+geindex+'_'+testindex[k]+'_'+feature+'.mat'
      testPaths.append(filePath)


  #获取训练集数据和标签
  for train_path in trainPaths:
    f = scio.loadmat(train_path)
    label = f['label'][0][0]-ge_first_index
    data = f['data'].astype(np.float32)
    x=0
    for x in range(data.shape[0]):
      if cflag:
        sigData=get_sig_img(data[x],sigmig_index)
      else:
        sigData=data[x]
      trainData.append(sigData)
      trainLabel.append(label)
  # 获取测试集数据和标签
  for test_path in testPaths:
    f = scio.loadmat(test_path)
    label = f['label'][0][0] - ge_first_index
    data = f['data'].astype(np.float32)
    for x in range(data.shape[0]):
      if cflag:
        sigData = get_sig_img(data[x], sigmig_index)
      else:
        sigData = data[x]
      testData.append(sigData)
      testLabel.append(label)

  trainDataArry = np.array(trainData, dtype=np.float32)
  testDataArry = np.array(testData, dtype=np.float32)
  trainLabelArry = np.array(trainLabel,dtype=np.int32)
  testLabelArry = np.array(testLabel,dtype=np.int32)

  trainDataArry = np.expand_dims(trainDataArry, axis=3)
  testDataArry = np.expand_dims(testDataArry, axis=3)
  trainDataArry = trainDataArry.transpose(0, 2, 3, 1)
  testDataArry = testDataArry.transpose(0, 2, 3, 1)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry,testDataArry,trainLabelArry,testLabelArry

def INiaPro_feature_single_test(datapath,ninapro,subject,classes,testindex,feature):

  if ninapro=='db5':
    # 训练集 第1、3,4,6,8,9,10
    channel=16
    ge_first_index=0
  elif ninapro=='db1':
    channel = 10
    ge_first_index = 1
  elif ninapro=='db2':
    # 训练集 第1、3,4,6,8,9,10
    channel=12
    ge_first_index=1
  elif ninapro=='db3':
    # 训练集 第1、3,4,6,8,9,10
    channel=12
    ge_first_index=0
  elif ninapro=='db7':
    # 训练集 第1、3,4,6,8,9,10
    channel=12
    ge_first_index=0

  testData = []
  #data/semg_data/ninapro_db1/001
  datapath=datapath+'/'+subject
  sigmig_index=genIndex(channel)
  geindex=str(classes+ge_first_index).rjust(3, '0')
  path1=datapath+'/'+geindex
  test_path=path1+'/'+subject+'_'+geindex+'_'+testindex+'_'+feature+'.mat'
  # 获取测试集数据和标签

  f = scio.loadmat(test_path)
  label = f['label'][0][0] - ge_first_index
  data = f['data'].astype(np.float32)
  for x in range(data.shape[0]):
    sigData = get_sig_img(data[x], sigmig_index)
    testData.append(sigData)
    # testLabel.append(label)
  testDataArry = np.array(testData, dtype=np.float32)
  testDataArry = np.expand_dims(testDataArry, axis=3)
  testDataArry = testDataArry.transpose(0, 2, 3, 1)
  return testDataArry



'''把多个特征合并成一个 合成维度为n*m*10  m=k1+k2+...
参数：f_merge 为列表，里面存多个特征 特征维度为n*k*10  
'''
def feature_merge(f_merge):
  mergeall = []
  for n in range(f_merge[0].shape[0]):
    merge = []
    for i in range(len(f_merge)):
      merge.append(f_merge[i][n])
    merge = np.concatenate(tuple(merge), axis=0)
    mergeall.append(merge)
  mergeall = np.array(mergeall)
  return mergeall
def INiaPro_feature_multi_test(datapath,ninapro,subject,classes,testindex,fList):

  if ninapro=='db5':
    # 训练集 第1、3,4,6,8,9,10
    channel=16
    ge_first_index=0
  elif ninapro=='db1':
    channel = 10
    ge_first_index = 1
  elif ninapro=='db2':
    channel=12
    ge_first_index=1
  elif ninapro=='db3':
    # 训练集 第1、3,4,6,8,9,10
    channel=12
    ge_first_index=0
  elif ninapro=='db7':
    # 训练集 第1、3,4,6,8,9,10
    channel=12
    ge_first_index=0
  testData = []
  #data/semg_data/ninapro_db1/001
  datapath=datapath+'/'+subject
  sigmig_index = genIndex(channel)
  #遍历手势

  geindex = str(classes+ge_first_index).rjust(3, '0')
  path1=datapath+'/'+geindex
  f_merge=[]
  for x in range(len(fList)):
    filePath=path1+'/'+subject+'_'+geindex+'_'+testindex+'_'+fList[x]+'.mat'
    data = scio.loadmat(filePath)['data']
    f_merge.append(data)
  mergeallArry=feature_merge(f_merge)
  #添加标签
  for k in range(mergeallArry.shape[0]):
    sigData = get_sig_img(mergeallArry[k], sigmig_index)
    testData.append(sigData)

  testDataArry = np.array(testData, dtype=np.float32)
  testDataArry = np.expand_dims(testDataArry, axis=3)
  testDataArry = testDataArry.transpose(0, 2, 3, 1)
  return testDataArry

def INiaPro_feature_split_multi(datapath,subject,fList,cflag,ninapro):

  if ninapro=='db5':
    # 训练集 第1、3,4,6,8,9,10
    trainindex = ['000', '002', '003', '005']
    testindex = ['001', '004']
    classes=41
    channel=16
    ge_first_index=0
  elif ninapro=='db1':
    trainindex = ['000', '002', '003', '005', '007', '008', '009']
    testindex = ['001', '004', '006']
    classes = 52
    channel = 10
    ge_first_index = 1
  elif ninapro == 'db7':
    # 训练集 第1、3,4,6,8,9,10
    trainindex = ['000', '002', '003', '005']
    testindex = ['001', '004']
    classes = 41
    channel = 12
    ge_first_index = 0
  trainData = []
  testData = []
  trainLabel = []
  testLabel = []
  datapath=datapath+'/'+subject
  sigmig_index = genIndex(channel)
  #遍历手势
  for i in tqdm(range(classes)):
    # geindex=str(i+1).rjust(3, '0')
    geindex = str(i+ge_first_index).rjust(3, '0')
    path1=datapath+'/'+geindex
    #遍历训练集
    for j in range(len(trainindex)):
      f_merge=[]
      for x in range(len(fList)):
        filePath=path1+'/'+subject+'_'+geindex+'_'+trainindex[j]+'_'+fList[x]+'.mat'
        data = scio.loadmat(filePath)['data']
        f_merge.append(data)
      mergeallArry=feature_merge(f_merge)
      #添加标签
      for k in range(mergeallArry.shape[0]):
        trainLabel.append(i)
        if cflag:
          sigData = get_sig_img(mergeallArry[k], sigmig_index)
        else:
          sigData = mergeallArry[k]
        trainData.append(sigData)

    #遍历生成测试集
    for j in range(len(testindex)):
      f_merge = []
      for x in range(len(fList)):
        filePath = path1 + '/' + subject + '_' + geindex + '_' + testindex[j] + '_' + fList[x] + '.mat'
        data = scio.loadmat(filePath)['data']
        f_merge.append(data)
      mergeallArry = feature_merge(f_merge)
      # 添加标签
      for k in range(mergeallArry.shape[0]):
        testLabel.append(i)
        if cflag:
          sigData = get_sig_img(mergeallArry[k], sigmig_index)
        else:
          sigData = mergeallArry[k]
        testData.append(sigData)

  trainDataArry = np.array(trainData, dtype=np.float32)
  testDataArry = np.array(testData, dtype=np.float32)
  trainLabelArry = np.array(trainLabel)
  testLabelArry = np.array(testLabel)
  trainDataArry = np.expand_dims(trainDataArry, axis=3)
  testDataArry = np.expand_dims(testDataArry, axis=3)
  trainDataArry = trainDataArry.transpose(0, 2, 3, 1)
  testDataArry = testDataArry.transpose(0, 2, 3, 1)
  # ------------------------------------------------------#
  # 转为二值类别矩阵
  trainLabelArry = tf.keras.utils.to_categorical(trainLabelArry)
  testLabelArry = tf.keras.utils.to_categorical(testLabelArry)
  return trainDataArry,testDataArry,trainLabelArry,testLabelArry



expected_features_withimu = {
  "X1": tf.io.FixedLenFeature([], dtype=tf.string),
  "X2": tf.io.FixedLenFeature([], dtype=tf.string),
  "X3": tf.io.FixedLenFeature([], dtype=tf.string),
  "X4": tf.io.FixedLenFeature([], dtype=tf.string),
  "Y1": tf.io.FixedLenFeature([], dtype=tf.string)
}
expected_features = {
  "X1": tf.io.FixedLenFeature([], dtype=tf.string),
  "X2": tf.io.FixedLenFeature([], dtype=tf.string),
  "X3": tf.io.FixedLenFeature([], dtype=tf.string),
  "Y1": tf.io.FixedLenFeature([], dtype=tf.string)
}
expected_single = {
  "X1": tf.io.FixedLenFeature([], dtype=tf.string),
  "Y1": tf.io.FixedLenFeature([], dtype=tf.string)
}

def parse_example(serialized_example,ninapro):
  example = tf.io.parse_single_example(serialized_example,
                                       expected_features)
  X1 = tf.io.decode_raw(example["X1"], out_type=tf.float32)
  X2 = tf.io.decode_raw(example["X2"], out_type=tf.float32)
  X3 = tf.io.decode_raw(example["X3"], out_type=tf.float32)
  Y1 = tf.io.decode_raw(example["Y1"], out_type=tf.float32)
  if ninapro=='db1':
    X1 = tf.reshape(X1, [50, 1, 32])
    X2 = tf.reshape(X2, [50, 1, 22])
    X3 = tf.reshape(X3, [50, 1, 28])
    Y1 = tf.reshape(Y1, [52])
  elif ninapro=='db5':
    X1 = tf.reshape(X1, [128, 1, 64])
    X2 = tf.reshape(X2, [128, 1, 42])
    X3 = tf.reshape(X3, [128, 1, 48])
    Y1 = tf.reshape(Y1, [41])
  elif ninapro == 'db7':
    X1 = tf.reshape(X1, [72, 1, 32])
    X2 = tf.reshape(X2, [72, 1, 22])
    X3 = tf.reshape(X3, [72, 1, 28])
    Y1 = tf.reshape(Y1, [41])
  return {'inputx0':X1,'inputx1':X2,'inputx2':X3},{'output':Y1}

def parse_example_single(serialized_example,ninapro):
  example = tf.io.parse_single_example(serialized_example,
                                       expected_single)
  X1 = tf.io.decode_raw(example["X1"], out_type=tf.float32)

  Y1 = tf.io.decode_raw(example["Y1"], out_type=tf.float32)
  if ninapro=='db1':
    X3 = tf.reshape(X1, [50, 28, 1])
    Y1 = tf.reshape(Y1, [52])
  elif ninapro=='db5':
    X1 = tf.reshape(X1, [128, 1, 64])
    # X2 = tf.reshape(X2, [128, 1, 42])
    # X3 = tf.reshape(X3, [128, 1, 48])
    Y1 = tf.reshape(Y1, [41])
  elif ninapro=='db7':
    X1 = tf.reshape(X1, [72, 1, 32])
    # X3 = tf.reshape(X3, [72, 1, 28])
    Y1 = tf.reshape(Y1, [41])
  return {'inputx0':X1},{'output':Y1}




def parse_example_withimu(serialized_example,ninapro):
  example = tf.io.parse_single_example(serialized_example,
                                       expected_features_withimu)
  X1 = tf.io.decode_raw(example["X1"], out_type=tf.float32)
  X2 = tf.io.decode_raw(example["X2"], out_type=tf.float32)
  X3 = tf.io.decode_raw(example["X3"], out_type=tf.float32)
  X4 = tf.io.decode_raw(example["X4"], out_type=tf.float32)
  Y1 = tf.io.decode_raw(example["Y1"], out_type=tf.float32)
  if ninapro=='db1':
    X1 = tf.reshape(X1, [50, 1, 32])
    X2 = tf.reshape(X2, [50, 1, 22])
    X3 = tf.reshape(X3, [50, 1, 28])
    Y1 = tf.reshape(Y1, [52])
  elif ninapro=='db5':
    X1 = tf.reshape(X1, [128, 1, 64])
    X2 = tf.reshape(X2, [128, 1, 42])
    X3 = tf.reshape(X3, [128, 1, 48])
    X4 = tf.reshape(X4, [3, 1, 40])
    Y1 = tf.reshape(Y1, [41])
  elif ninapro=='db7':
    X1 = tf.reshape(X1, [72, 1, 32])
    X2 = tf.reshape(X2, [72, 1, 22])
    X3 = tf.reshape(X3, [72, 1, 28])
    X4 = tf.reshape(X4, [36, 1, 20])
    Y1 = tf.reshape(Y1, [41])

  return {'inputx0':X1,'inputx1':X2,'inputx2':X3,'inputx3':X4},{'output':Y1}

def tfrecords_reader_dataset_single(fileList,shuffle_buffer_size,batch_size,ninapro,imuFlag,epoch, n_readers=1,
                             n_parse_threads=5
                             ):
  dataset = tf.data.Dataset.list_files(fileList)
  dataset = dataset.interleave(
    lambda filename: tf.data.TFRecordDataset(
      filename),
    cycle_length=n_readers
  )
  dataset = dataset.map(lambda x:parse_example_single(x,ninapro),
                        num_parallel_calls=n_parse_threads)
  dataset = dataset.shuffle(shuffle_buffer_size,seed=666)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(epoch)
  return dataset


def tfrecords_reader_dataset(fileList,shuffle_buffer_size,batch_size,ninapro,imuFlag,epoch, n_readers=1,
                             n_parse_threads=5
                             ):
  dataset = tf.data.Dataset.list_files(fileList)
  dataset = dataset.interleave(
    lambda filename: tf.data.TFRecordDataset(
      filename),
    cycle_length=n_readers
  )

  if imuFlag:
    dataset = dataset.map(lambda x: parse_example_withimu(x, ninapro),
                          num_parallel_calls=n_parse_threads)
  else:
    dataset = dataset.map(lambda x:parse_example(x,ninapro),
                        num_parallel_calls=n_parse_threads)
  dataset = dataset.shuffle(shuffle_buffer_size,seed=666)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(epoch)
  return dataset


def countTfRecord(filepath):
    count=0
    for record in tf.compat.v1.io.tf_record_iterator(filepath):
      count+=1
    print('数据{0}的数量是{1}'.format(filepath, count))
    return count


def getDataDictCount(filePath):
  myDict=np.load(filePath,allow_pickle='TRUE').item()
  countSum=sum(myDict.values())
  return countSum

def GetTfFilesList(mode,ninapro):
  fileList=[]
  if ninapro=='db1':
    subNum=27
  elif ninapro=='db5':
    subNum=10
  elif ninapro=='db7':
    subNum=21
  for i in range(subNum):
    geindex = str(i).rjust(3, '0')
    files='data/pretrain_emg+imu/'+ninapro+'/'+mode+'_'+geindex+'.tfrecords'
    fileList.append(files)
  return fileList
if __name__ == '__main__':
  print('111')

