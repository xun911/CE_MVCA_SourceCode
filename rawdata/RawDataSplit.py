import scipy.io as scio
import numpy as np
import os
from imutils import paths
import pandas as pd
from utils import save_excel
from scipy import signal

def SaveMat(data,subject,gesture,trial):
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

def Deal_Ge0(arrAll,subject):
    rep0=[]
    rep1=[]
    rep2=[]
    rep3=[]
    rep4=[]
    rep5=[]
    for arr in arrAll:
        rep0.append(arr[0])
        rep1.append(arr[1])
        rep2.append(arr[2])
        rep3.append(arr[3])
        rep4.append(arr[4])
        rep5.append(arr[5])
    rep0=np.vstack(rep0)
    rep1 = np.vstack(rep1)
    rep2 = np.vstack(rep2)
    rep3 = np.vstack(rep3)
    rep4 = np.vstack(rep4)
    rep5 = np.vstack(rep5)
    SaveMat(rep0,subject-1,0,0)
    SaveMat(rep1, subject-1, 0, 1)
    SaveMat(rep2, subject-1, 0, 2)
    SaveMat(rep3, subject-1, 0, 3)
    SaveMat(rep4, subject-1, 0, 4)
    SaveMat(rep5, subject-1, 0, 5)


#db2 db3 minlabel 1 18 41
def RawDataSplit_single2(filePath,subject,mode,minlabel):
    data = scio.loadmat(filePath)
    if mode == 'imu':
        emg = data['acc']
    elif mode == 'emg':
        emg = data['emg']
    elif mode == 'glove':
        emg = data['glove']
    label = data['restimulus']
    rep = data['rerepetition']
    dfData = pd.DataFrame(emg)
    df=np.hstack([label,rep])
    df = pd.DataFrame(df)
    df.columns = ['label', 'rep']

    maxLabel=max(label)[0]
    minLabel=minlabel
    maxRep=max(rep)[0]

    arrAll=[]
    labelList=list(range(minLabel,maxLabel+1))
    labelList.append(0)
    for i in labelList:
        for j in range(1,maxRep+1):
            selectIndex = df[(df["label"] == i) & (df["rep"] == j)].index.values
            selectData = dfData.iloc[selectIndex, :].values
            if i==0:
                arrAll.append(selectData)
            else:
                SaveMat(selectData,subject-1,i,j-1)
    return arrAll

def RawDataSplit_single(filePath,subject,mode,baseNum):
    data = scio.loadmat(filePath)
    if mode == 'imu':
        emg = data['acc']
    elif mode == 'emg':
        emg = data['emg']
    elif mode == 'glove':
        emg = data['glove']
    label = data['restimulus']
    rep = data['rerepetition']
    dfData = pd.DataFrame(emg)
    df=np.hstack([label,rep])
    df = pd.DataFrame(df)
    df.columns = ['label', 'rep']
    maxLabel=max(label)[0]
    minLabel=min(label)[0]
    maxRep=max(rep)[0]
    minRep=min(rep)[0]
    arrAll=[]
    for i in range(0,maxLabel+1):
        for j in range(1,maxRep+1):
            selectIndex = df[(df["label"] == i) & (df["rep"] == j)].index.values
            selectData = dfData.iloc[selectIndex, :].values
            if i==0:
                arrAll.append(selectData)
            else:
                SaveMat(selectData,subject-1,i+baseNum,j-1)
    return arrAll

def RawDataSplit_subject(subPath,subject,mode,ninapro):
    mat_paths = sorted(list(paths.list_files(subPath)))
    if ninapro=='db5':
        arrAll_0 = RawDataSplit_single(mat_paths[0], subject, mode, 0)
        arrAll_1 = RawDataSplit_single(mat_paths[1], subject, mode, 0)
        arrAll_2 = RawDataSplit_single(mat_paths[2], subject, mode, 17)
        arrAll = np.vstack([arrAll_0,arrAll_1, arrAll_2])
    elif ninapro=='db1':
        arrAll_0 = RawDataSplit_single(mat_paths[0], subject, mode, 0)
        arrAll_1 = RawDataSplit_single(mat_paths[1], subject, mode, 12)
        arrAll_2 = RawDataSplit_single(mat_paths[2], subject, mode, 29)
        arrAll = np.vstack([arrAll_0, arrAll_1, arrAll_2])
    elif (ninapro=='db2')|(ninapro=='db3'):
        arrAll_0 = RawDataSplit_single2(mat_paths[0], subject, mode, 1)
        arrAll_1 = RawDataSplit_single2(mat_paths[1], subject, mode, 18)
        arrAll_2 = RawDataSplit_single2(mat_paths[2], subject, mode, 41)
        arrAll = np.vstack([arrAll_0, arrAll_1, arrAll_2])
    elif ninapro=='db7':
        arrAll_0 = RawDataSplit_single2(mat_paths[0], subject, mode, 1)
        arrAll_1 = RawDataSplit_single2(mat_paths[1], subject, mode, 18)
        arrAll = np.vstack([arrAll_0, arrAll_1])
    if ninapro !='db1':
        Deal_Ge0(arrAll, subject)

def Compare(filePath1,filePath2):
    mat_paths1 = sorted(list(paths.list_files(filePath1)))
    mat_paths2 = sorted(list(paths.list_files(filePath2)))
    for i in range(len(mat_paths1)):
        print(mat_paths1[i])
        data1=scio.loadmat(mat_paths1[i])['data']
        data2 = scio.loadmat(mat_paths2[i])['data']
        try:
            print((data1==data2).all())
        except:
            print('error')
def GetMatTotalNum(filePath):
    num=0
    mat_paths= sorted(list(paths.list_files(filePath)))
    for i in range(len(mat_paths)):
        data=scio.loadmat(mat_paths[i])['data']
        num+=data.shape[0]
    return num

def GetRawDataIndex(dataPath,seLabel,seRep):
    data = scio.loadmat(dataPath)
    emg = data['emg']
    label = data['restimulus']
    rep = data['rerepetition']
    dfData = pd.DataFrame(emg)
    df = np.hstack([label, rep])
    df = pd.DataFrame(df)
    df.columns = ['label', 'rep']
    selectIndex = df[(df["label"] == seLabel) & (df["rep"] == seRep)].index.values
    selectData = dfData.iloc[selectIndex, :].values
    return selectIndex,selectData
    print('111')



if __name__ == '__main__':
    ninapro = 'db1'
    model='emg'
    outputPath = '../data/ninapro_'+ninapro
    if model=='imu':
        outputPath = '../data/imu/' + ninapro

    if ninapro=='db3':
        subjects =[2,4,5,6,7,8,9,11]
    elif ninapro=='db2':
        subjects =range(1,41)
    elif ninapro=='db5':
        subjects =range(1,11)
    elif ninapro=='db1':
        subjects = range(1, 28)
    elif ninapro == 'db7':
        subjects = range(1, 23)
        subjects.remove(21)
    for i in subjects:
        print('subject{0}--------'.format(i-1))
        filePath='../rawdata/ninapro_'+ninapro+'/s'+str(i)
        RawDataSplit_subject(filePath,i,model,ninapro)
    print('111')
