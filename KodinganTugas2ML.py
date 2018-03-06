#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:11:45 2018

@author: VidoValianto
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

bias= 0.7
alpha= 0.1
mdteta = np.zeros(shape=(4,1))
mtetab = np.zeros(shape=(4,1))
mtotalerror= np.zeros(shape=(100,1))
mtotalerror2= np.zeros(shape=(100,1))
mtotalfullerror= np.zeros(shape=(60,1))
mtotalfullerror2= np.zeros(shape=(60,1))
mtotalfullerrortrain= np.zeros(shape=(60,1))
mtotalfullerrorvalid= np.zeros(shape=(60,1))

dataframe = pd.read_csv("./data.csv")
dataframe = dataframe['teta'].str.split(',', expand=True)
d = {'teta1': [0.3], 'teta2': [0.6], 'teta3': [0.9], 'teta4': [0.2]}

teta = pd.DataFrame(data=d)

dataframe[4] = dataframe[4].str.replace('Iris-setosa', '1')
dataframe[4] = dataframe[4].str.replace('Iris-versicolor', '0')

datatrain1= dataframe[dataframe[4] == '1']
datatrain2= dataframe[dataframe[4] == '0']

datavalidasi1 = datatrain1.iloc[40:,:]
datavalidasi2 = datatrain2.iloc[40:,:]

datatrain1 = datatrain1.iloc[:40,:]
datatrain2 = datatrain2.iloc[:40,:]

datatrain = pd.concat([datatrain1,datatrain2])
datavalidasi = pd.concat([datavalidasi1,datavalidasi2])

dataframe = dataframe.astype('float64')
datatrain = datatrain.astype('float64')
datavalidasi = datavalidasi.astype('float64')
teta = teta.astype('float64')

def perhitunganxteta(x):
    return (np.dot(dataframe.iloc[x, :-1],np.transpose(teta)))

def h(x,bias):
    return (x+bias)

def sigm(hasilh):
   return (1/(1+math.exp( -hasilh )))
   
   
def perhitungansigmsaturow(x,bias):
    return sigm(h(perhitunganxteta(x),bias))
   
def error(kelas,sigm):
    return ((kelas - sigm)**2)
    
def prediksi(msigm):
    if msigm < 0.5:
        return 0
    else:
        return 1
    
def perhitungandeltax(sigmoid,kelas,x,y):
    return (2*(sigmoid- kelas)*(1-sigmoid)*sigmoid*dataframe.iloc[x, y])

def perhitungandeltabias(sigmoid,kelas):
    return (2*(sigmoid- kelas)*(1-sigmoid)*sigmoid)

def tetabaru(teta,alpha,deltateta):
    return (teta-(alpha*deltateta))
  
def biasbaru(bias,alpha,deltabias):
    return (bias-(alpha*deltabias))

def gantibias(mbias):
    bias = mbias.copy()
    
def jumlahtotalerrorepoch(merror,x):
    mtotalerror[x]=merror.copy()
    
def jumlahtotalerrorepoch2(merror2,x):
    mtotalerror2[x]=merror2.copy()

def jumlahtotalerrorfull(totalerrortiapepoch,x):
    mtotalfullerror[x]=totalerrortiapepoch.copy()
    
def jumlahtotalerrorfull2(totalerrortiapepoch2,x):
    mtotalfullerror2[x]=totalerrortiapepoch2.copy()


def perhitunganepoch(dataframe,datavalidasi):
    for i in range(len(dataframe)):
        msigm = perhitungansigmsaturow(i,bias)
        merror = error(dataframe.iloc[i, 4],msigm)
        mprediksi = prediksi(msigm)
        mdbias = perhitungandeltabias(msigm,dataframe.iloc[i, 4])
        print (" TRAIN sigmoid = ",msigm, " error = ", merror, " prediksi = ", mprediksi)
        y = 0
        while(y < len(teta.columns)):
            mdteta[y] = perhitungandeltax(msigm,dataframe.iloc[i, 4],i,y)
            mtetab[y] = tetabaru(teta.iloc[0, y],alpha,mdteta[y])
            y = y+1
        y = 0
        while(y < len(teta.columns)):
            teta.iloc[0, y]= mtetab[y].copy()
            y = y+1
        mbiasb = biasbaru(bias,alpha,mdbias)
        gantibias(mbiasb)
        jumlahtotalerrorepoch(merror,i)
        
        msigm2 = perhitungansigmsaturow(i,bias)
        merror2 = error(dataframe.iloc[i, 4],msigm2)
        mprediksi2 = prediksi(msigm2)
        print (" VALID sigmoid = ",msigm2, " error = ", merror2, " prediksi = ", mprediksi2)
        jumlahtotalerrorepoch2(merror2,i)


def perhitunganfull(berapakali,data,datavalid):
    
    for i in range(berapakali):
        perhitunganepoch(data,datavalid)
        jumlahtotalerrorfull(sum(mtotalerror)/len(mtotalerror),i)
        jumlahtotalerrorfull2(sum(mtotalerror2)/len(mtotalerror2),i)
        mtotalfullerrortrain = mtotalfullerror.copy()
        mtotalfullerrorvalid = mtotalfullerror2.copy()
        
    plt.plot(mtotalfullerrortrain)
    plt.plot(mtotalfullerrorvalid,color="green")
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.show()


perhitunganfull(60,datatrain,datavalidasi)