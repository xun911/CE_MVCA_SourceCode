import gc

from tqdm import tqdm
import tensorflow as tf
import dataloader
import numpy as np
#获得模型预训练的数据集



def DataToTFrecord_single(ninapro):
  # subjectList = ['000']
  # ninapro=data_path.split('_')[1]
  trainCountDict = {}
  testCountDict = {}
  if ninapro=='db1':
    data_path = '../../extract_features/out_features/ninapro-db1-var-raw-prepro-lowpass-win-20-stride-1'
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                 '014', '015', '016', '017', '018', '019', '020', '021',
                 '022', '023', '024', '025', '026']
    # subjectList = ['003']
  elif ninapro=='db5':
    data_path = '../../extract_features/out_features/ninapro-db5-var-raw-prepro-lowpass-win-40-stride-20'
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    # subjectList = ['000']

  elif ninapro=='db7':
    data_path = '../../extract_features/out_features/ninapro-db7-downsample20-var-raw-prepro-lowpass-win-20-stride-1'
    subjectList = ['000', '001', '002', '003','004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                   '014', '015', '016', '017', '018', '019', '021']
  else:
      print('please check data path')
      return
  for i in tqdm(range(len(subjectList))):
    print('the {0} ------------'.format(i))
    trainPath='../../data/pretrain/'+ninapro+'/train_'+subjectList[i]+'.tfrecords'
    testPath = '../../data/pretrain/'+ninapro+'/test_' + subjectList[i] + '.tfrecords'
    writer_train = tf.io.TFRecordWriter(trainPath)
    writer_test = tf.io.TFRecordWriter(testPath)

    preData1 = dataloader.INiaPro_feature_split_single(data_path, subjectList[i], 'dwpt', True, ninapro)
    preData2 = dataloader.INiaPro_feature_split_single(data_path, subjectList[i], 'dwt', True,ninapro)
    preData3 = dataloader.INiaPro_feature_split_multi(data_path, subjectList[i],
                                                          ['mav', 'wl', 'wamp', 'mavslpframewise', 'arc',
                                                           'mnf_MEDIAN_POWER', 'psr'], True,ninapro)
    for j in range(preData1[0].shape[0]):
      example_train = tf.train.Example(features=tf.train.Features(feature={
        "X1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[0][j].astype(np.float32).tostring()])),
        "X2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData2[0][j].astype(np.float32).tostring()])),
        "X3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData3[0][j].astype(np.float32).tostring()])),
        "Y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[2][j].astype(np.float32).tostring()]))
      }))
      writer_train.write(example_train.SerializeToString())  # 序列化为字符串
    for k in range(preData1[1].shape[0]):
      example_test = tf.train.Example(features=tf.train.Features(feature={
        "X1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[1][k].astype(np.float32).tostring()])),
        "X2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData2[1][k].astype(np.float32).tostring()])),
        "X3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData3[1][k].astype(np.float32).tostring()])),
        "Y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[3][k].astype(np.float32).tostring()]))
      }))
      writer_test.write(example_test.SerializeToString())  # 序列化为字符串
    writer_train.close()
    writer_test.close()
    trainCountDict[subjectList[i]] = preData1[0].shape[0]
    testCountDict[subjectList[i]] = preData1[1].shape[0]

    del preData1
    del preData2
    del preData3
    gc.collect()
  np.save('../../data/pretrain/' + ninapro + '/trainCount.npy', trainCountDict)
  np.save('../../data/pretrain/' + ninapro + '/testCount.npy', testCountDict)

  print('-------------------sucess!-------------------')
def DataToTFrecord_single_withimu(ninapro):
  trainCountDict = {}
  testCountDict = {}

  imuPath='../../data/imu/'+str(ninapro)
  if ninapro=='db1':
    data_path = '../../extract_features/out_features/ninapro-db1-var-raw-prepro-lowpass-win-20-stride-1'
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                 '014', '015', '016', '017', '018', '019', '020', '021',
                 '022', '023', '024', '025', '026']
  elif ninapro=='db5':
    data_path = '../../extract_features/out_features/ninapro-db5-var-raw-prepro-lowpass-win-40-stride-20'
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']

  elif ninapro=='db7':
    data_path = '../../extract_features/out_features/ninapro-db7-downsample20-var-raw-prepro-lowpass-win-20-stride-1'
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                   '014', '015', '016', '017', '018', '019', '021']
  else:
      print('please check data path')
      return
  for i in tqdm(range(len(subjectList))):
    print('the {0} ------------'.format(subjectList[i]))
    trainPath='../../data/pretrain/'+ninapro+'/train_'+subjectList[i]+'.tfrecords'
    testPath = '../../data/pretrain/'+ninapro+'/test_' + subjectList[i] + '.tfrecords'
    writer_train = tf.io.TFRecordWriter(trainPath)
    writer_test = tf.io.TFRecordWriter(testPath)

    preData1 = dataloader.INiaPro_feature_split_single(data_path, subjectList[i], 'dwpt', True,ninapro)
    preData2 = dataloader.INiaPro_feature_split_single(data_path, subjectList[i], 'dwt', True,ninapro)
    preData3 = dataloader.INiaPro_feature_split_multi(data_path, subjectList[i],
                                                          ['mav', 'wl', 'wamp', 'mavslpframewise', 'arc',
                                                           'mnf_MEDIAN_POWER', 'psr'], True,ninapro)

    preData4 = dataloader.INiaPro_imu(imuPath, subjectList[i], ninapro)
    for j in range(preData1[0].shape[0]):
      example_train = tf.train.Example(features=tf.train.Features(feature={
        "X1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[0][j].astype(np.float32).tostring()])),
        "X2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData2[0][j].astype(np.float32).tostring()])),
        "X3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData3[0][j].astype(np.float32).tostring()])),
        "X4": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData4[0][j].astype(np.float32).tostring()])),
        "Y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[2][j].astype(np.float32).tostring()]))
      }))
      writer_train.write(example_train.SerializeToString())  # 序列化为字符串
    for k in range(preData1[1].shape[0]):
      example_test = tf.train.Example(features=tf.train.Features(feature={
        "X1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[1][k].astype(np.float32).tostring()])),
        "X2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData2[1][k].astype(np.float32).tostring()])),
        "X3": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData3[1][k].astype(np.float32).tostring()])),
        "X4": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData4[1][k].astype(np.float32).tostring()])),
        "Y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[preData1[3][k].astype(np.float32).tostring()]))
      }))
      writer_test.write(example_test.SerializeToString())  # 序列化为字符串
    writer_train.close()
    writer_test.close()
    #tfrecord count
    trainCountDict[subjectList[i]] =preData1[0].shape[0]
    testCountDict[subjectList[i]] =preData1[1].shape[0]

    del preData1
    del preData2
    del preData3
    del preData4
    gc.collect()
  np.save('../../data/pretrain/' + ninapro + '/trainCount.npy', trainCountDict)
  np.save('../../data/pretrain/' + ninapro + '/testCount.npy', testCountDict)
  print('-------------------sucess!-------------------')
def dataCountToDict(ninapro):
  if ninapro == 'db1':
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                   '014', '015', '016', '017', '018', '019', '020', '021',
                   '022', '023', '024', '025', '026']
  elif ninapro == 'db5':
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']
  elif ninapro == 'db7':
    subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                   '014', '015', '016', '017', '018', '019', '021']
  else:
    print('please check data path')
    return
  trainCountDict={}
  testCountDict = {}
  for i in range(len(subjectList)):
    print('the {0} ------------'.format(i))
    trainPath='../../data/pretrain/'+ninapro+'/train_'+subjectList[i]+'.tfrecords'
    trainCountDict[subjectList[i]]=countTfRecord(trainPath)
    testPath = '../../data/pretrain/'+ninapro+ '/test_' + subjectList[i] + '.tfrecords'
    testCountDict[subjectList[i]] = countTfRecord(testPath)

  np.save('../../data/pretrain/'+ninapro+'/trainCount.npy',trainCountDict)
  np.save('../../data/pretrain/'+ninapro+'/testCount.npy', testCountDict)

def countTfRecord(filepath):
    count = 0
    for record in tf.compat.v1.io.tf_record_iterator(filepath):
      count += 1
    print('数据{0}的数量是{1}'.format(filepath, count))
    return count

if __name__ == '__main__':
    DataToTFrecord_single('db1')
    # DataToTFrecord_single_withimu('db7')
