import gc

from utils import Find_Majority,save_excel,GetFCScore
import numpy as np
from tensorflow.keras.models import load_model
from statistics import mean
import os
import argparse
import tensorflow as tf
import time
from math import log
from dataloader import INiaPro_feature_multi_test,INiaPro_feature_single_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from numba import cuda
def Set_GPU(gIndex):
    os.environ["CUDA_VISIBLE_DEVICES"] = gIndex #指定第一块GPU可用
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        #设置显存不占满
        tf.config.experimental.set_memory_growth(gpu, True)
        #设置显存占用最大值
        tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
        )


def ITR(n, p, t):
    if not all(isinstance(i, (int, float)) for i in [n, p, t]):
        raise TypeError("Inputs must be numeric.")
    if p < 0 or p > 1:
        raise ValueError("Accuracy needs to be between 0 and 1.")
    elif p < 1/n:
        print("Warning: The ITR might be incorrect because the accuracy < chance level.")
        itr = 0
    elif p == 1:
        itr = math.log2(n) * 60 / t
    else:
        itr = (math.log2(n) + p*math.log2(p) + (1-p)*math.log2((1-p)/(n-1))) * 60 / t
    return itr

def Vote_subject(ninapro, subject, modelPre):

    if ninapro == 'db5':
        testindex = ['001', '004']
        classes = 41
        data_path = 'extract_features/out_features/ninapro-db5-var-raw-prepro-lowpass-win-40-stride-20'
    elif ninapro == 'db7':
        testindex = ['001', '004']
        classes = 41
        data_path = 'extract_features/out_features/ninapro-db7-downsample20-var-raw-prepro-lowpass-win-20-stride-1'
    elif ninapro == 'db1':
        testindex = ['001', '004', '006']
        data_path = 'extract_features/out_features/ninapro-db1-var-raw-prepro-lowpass-win-20-stride-1'
        classes = 52
        # 生成训练集和测试集路径
    totalNum = classes * len(testindex)
    trueNum = 0
    start = time.time()
    # matNums=0
    for i in range(classes):
        # geindex=str(i+1).rjust(3, '0')
        # 生成训练集文件路径
        for j in range(len(testindex)):
            # load data
            inputx0=INiaPro_feature_single_test(data_path,ninapro,subject,i,testindex[j],'dwpt')
            inputx1 = INiaPro_feature_single_test(data_path, ninapro, subject, i, testindex[j], 'dwt')
            inputx2 = INiaPro_feature_multi_test(data_path, ninapro, subject, i, testindex[j], ['mav', 'wl', 'wamp', 'mavslpframewise', 'arc',
                                                           'mnf_MEDIAN_POWER', 'psr'])
            preY = modelPre([inputx0,inputx1,inputx2])
            preY_label = [np.argmax(one_hot) for one_hot in preY]
            preLabel = Find_Majority(preY_label)
            if preLabel == i:
                trueNum += 1
    voteAcc = trueNum / totalNum
    end = time.time()
    fTime = end - start
    itrTime = fTime / totalNum
    itr = ITR(classes, voteAcc, itrTime)

    return voteAcc, itr, itrTime

if __name__ == '__main__':
    # itr=ITR(6,0.8975,0.10)

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--ninapro', default='db7', choices=['db1', 'db5','db7'],
                    help='select ninpro data')
    ap.add_argument('-g', '--GPUIndex', default='1', choices=['0', '1'],
                    help='select ldle GPU')
    args = vars(ap.parse_args())
    #GPU setting
    # Set_GPU(args['GPUIndex'])
    Set_GPU(args['GPUIndex'])
    # showMatrix_all('bio')
    ninapro = args['ninapro']
    if ninapro == 'db1':
        classes = 52
        subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                       '014', '015', '016', '017', '018', '019', '020', '021',
                       '022', '023', '024', '025', '026']
    elif ninapro == 'db7':
        subjectList = ['000', '001', '002', '003','004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                       '014', '015', '016', '017', '018', '019', '021']

    elif ninapro == 'db5':
        subjectList = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']
    aveVoteAcc=0
    voteAccList=[]
    accList=[]
    precisionList=[]
    recallList=[]
    f1List=[]
    itrList=[]
    itrTimeList=[]
    start = time.time()
    for i in tqdm(range(len(subjectList))):
        modelPre = load_model('model_my/{0}/{1}.h5'.format(ninapro, subjectList[i]))
        voteAcc, itr, itrTime = Vote_subject(ninapro, subjectList[i], modelPre)
        voteAccList.append(voteAcc)
        itrList.append(itr)
        itrTimeList.append(itrTime)
        del modelPre
        gc.collect()
    avgscore = mean(voteAccList)
    end = time.time()
    fTime = end - start
    #save and print
    outPath = 'output/' + ninapro + '/testAll_my.xls'
    save_excel([subjectList, voteAccList,itrList,itrTimeList],
               ['subject', 'voteAcc','itr','itrTime'], outPath)
    print('-------test all use {0} s'.format(fTime))
    print('Ninapro {0} average vote acc is {1}-------'.format(ninapro,avgscore))
