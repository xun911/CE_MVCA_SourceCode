
# 开发时间 2022/4/16 15:34
import os
import numpy as np
import pynvml
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from scipy import stats
import xlwt
import shutil
import pymrmr
import  pandas as pd

from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter

#返回一串1-10通道重排列之后的序号
def genIndex(chanums):
      index = []
      i = 1
      j = i+1

      if (chanums % 2) == 0:
         Ns = chanums+1
      else:
         Ns = chanums
      index.append(1)
      t = chr(i+ord('A'))
      while(i!=j):
          l = ""
          l = l+chr(i+ord('A'))
          l = l+chr(j+ord('A'))
          r = ""
          r = r+chr(j+ord('A'))
          r = r+chr(i+ord('A'))
          if(j>Ns):
              j = 1
          elif(t.find(l)==-1 and t.find(r)==-1):
              index.append(j)
              t = t+chr(j+ord('A'))
              i = j
              j = i+1
          else:
              j = j+1
      new_index = []
      if (chanums % 2) == 0:
          for i in range(len(index)):
              if index[i] != chanums+1:
                 new_index.append(index[i])
          index = new_index
      index = np.array(index)
      index = index-1
      return index
#获得通道重新排列的数据
def get_sig_img(data,sigmig_index):
    res=[]
    for sample in data:
        signal_img = sample[sigmig_index]
        signal_img = signal_img[:-1]
        # signal_img=np.array(signal_img)
        res.append(signal_img)
    res=np.array(res)
    # res1 =res.reshape(-1,sigmig_index.shape[0]-1)
    return res
def get_sig_img2(data, sigimg_index):
#     ch_num = data.shape[0]
#     sigimg_index = genIndex(ch_num)
     signal_img = data[sigimg_index]
     signal_img = signal_img[:-1]
#     print signal_img.shape
     return signal_img
#获取指定GPU剩余显存
def getGpuMrmory(id):
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('GPU{0}剩余{1}M'.format(id,meminfo.free/1024**2))



def CheckFolder(dataPath):
    flag=os.path.exists(dataPath)
    if flag!=True:
        os.makedirs(dataPath)
''' 该函数实现窗口宽度为七、滑动步长为1的滑动窗口截取序列数据 
参数：数据，窗口宽度，窗口步长，窗口起始值
'''
def sliding_window(data, sw_width, sw_step, in_start=0):
    # data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))  # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []

    for i in range(len(data)):
        in_end = in_start + sw_width
        # out_end = in_end + n_out
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；不然丢弃该样本
        if in_end <= len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end]

            X.append(train_seq)

        in_start += sw_step

    return np.array(X)



def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )
def get_segments_image(data, window, stride):
    chnum = data.shape[1];
    data=windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )
    data= data.reshape(-1, window, chnum)
    # data=data*255
    return data


def butter_lowpass_filter(data, cut, fs, order, zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt

    nyq = 0.5 * fs
    cut = cut / nyq

    b, a = butter(order, cut, btype='low')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y

def _get_sigimg_aux(data, sigimg_index):
    return np.transpose(get_sig_img(data.T, sigimg_index))

def get_sigimg1(data, sigimg_index):
        res = []
        for sample in data:
            amp=_get_sigimg_aux(sample, sigimg_index)
            res.append(amp[np.newaxis, ...])


        res = np.concatenate(res, axis=0)
#        res = res.reshape(res.shape[0], 1, res.shape[1], res.shape[2])
        res = res.reshape(res.shape[0], res.shape[1], res.shape[2], -1)
        return res
def downsample(data, step):
    return data[::step].copy()

def SaveMat(outputPath,data,subject,gesture,trial):
    #创建层级文件夹
    outputDir = os.path.join(
        outputPath,
        '{0:03d}',
        '{1:03d}').format(subject,gesture)
    if os.path.isdir(outputDir) is False:
        os.makedirs(outputDir)
    #保存mat文件
    out_path = os.path.join(
        outputDir,
        '{0:03d}_{1:03d}_{2:03d}.mat').format(subject, gesture,trial)
    scio.savemat(out_path,
                 {'data': data, 'label': gesture, 'subject': subject, 'trial': trial})
    print("Subject %d Gesture %d Trial %d saved!" % (subject, gesture, trial))

def save_excel(dataList,nameList,savePath):
    # 例如我们要存储两个list：name_list 和 err_list 到 Excel 两列
    # dataList = [[10, 20, 30],[0.99, 0.98, 0.97]]  # 示例数据
    # nameList = ['one', 'two', 'three']  # 示例数据
    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')
    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet1.write(0, 0, "序号")
    for i in range(len(nameList)):
        sheet1.write(0, i+1, nameList[i])  # 第1行第1列
        # sheet1.write(0, 1, "数量")  # 第1行第2列
        # sheet1.write(0, 2, "误差")  # 第1行第3列

    # 循环填入数据
    for i in range(len(dataList[0])):
        sheet1.write(i + 1, 0, i) # 第1列序号
        for j in range(len(dataList)):
            sheet1.write(i + 1, j+1, dataList[j][i])
            # sheet1.write(i + 1, 1, name_list[i])  # 第2列数量
        # sheet1.write(i + 1, 2, err_list[i])  # 第3列误差
    file.save(savePath)
    print('training record are saved in '+savePath)
def butter_filter(data,wLow,wHigh,fs,order,zero_phase=False):
    from scipy.signal import butter, lfilter, filtfilt
    nyq = 0.5 * fs
    # cut = cut / nyq
    # high = wHigh
    # low = wLow
    high=wHigh/nyq
    low=wLow/nyq
    b, a = butter(order, [low,high],'bandpass')
    y = (filtfilt if zero_phase else lfilter)(b, a, data)
    return y



'''获得数组中出现次数最多数字
'''
def Find_Majority(array):
    #找出数组中元素出现次数并按从大到小排序
    collection_words = Counter(array)
    #找出出现最多次数的
    most_counterNum = collection_words.most_common(1)
    mostwords=most_counterNum[0][0]
    return mostwords

if __name__ == '__main__':
    print('1111')
