#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:11:45 2018

@author: VidoValianto
"""
import pandas as pd
import numpy as np
import math

global hnya
global nsigm
global nerror
global tetab1
global tetab2
global tetab3
global tetab4
global biasb
mdteta = np.zeros(shape=(4,1))
mtetab = np.zeros(shape=(4,1))

dataframe = pd.read_csv("./data.csv")
dataframe = dataframe['teta'].str.split(',', expand=True)
d = {'teta1': [0.2], 'teta2': [0.1], 'teta3': [0.5], 'teta4': [0.3]}
bias = 0.2
alpha = 0.1
teta = pd.DataFrame(data=d)

dataframe[4] = dataframe[4].str.replace('Iris-setosa', '1')
dataframe[4] = dataframe[4].str.replace('Iris-versicolor', '0')

dataframe = dataframe.astype('double')
teta = teta.astype('double')

def perhitunganxteta(x):
    return (np.dot(dataframe.iloc[x, :-1],np.transpose(teta)))

def h(x,bias):
    return (x+bias)

def sigm(hasilh):
   return (1/(1+math.exp( -hasilh )))
   
   
def perhitungansigmsaturow(x,bias):
    nsigm = sigm(h(perhitunganxteta(x),bias))
    return sigm(h(perhitunganxteta(x),bias))
   
def error(kelas,sigm):
    return ((kelas - sigm)**2)
    
def prediksi(msigm):
    if msigm < 0.5:
        return 0
    else:
        return 1
    
def perhitungandeltax(sigmoid,kelas,x,y):
    print ("delta", x, y , "= " ,(2*(sigmoid- kelas)*(1-sigmoid)*sigmoid*dataframe.iloc[x, y]))
    return (2*(sigmoid- kelas)*(1-sigmoid)*sigmoid*dataframe.iloc[x, y])

def perhitungandeltabias(sigmoid,kelas):
    print ("delta bias" ,(2*(sigmoid- kelas)*(1-sigmoid)*sigmoid))
    return (2*(sigmoid- kelas)*(1-sigmoid)*sigmoid)

def tetabaru(teta,alpha,deltateta):
    print ("tetabaru" ,(teta-(alpha*deltateta)))
    return (teta-(alpha*deltateta))
  
def biasbaru(bias,alpha,deltabias):
    print ("biasbaru" ,(bias-(alpha*deltabias)))
    return (bias-(alpha*deltabias))

def gantiteta(tetabaru):
    teta = tetabaru

def gantiteta(biasbaru):
    biasb = tetabaru
    
def perhitunganepoch(dataframe):
    for i in range(len(dataframe)):
        msigm = perhitungansigmsaturow(i,bias)
        merror = error(dataframe.iloc[i, 4],msigm)
        mprediksi = prediksi(msigm)
        mdbias = perhitungandeltabias(msigm,dataframe.iloc[i, 4])
        print ("sigmoid = ",msigm, " error = ", merror, " prediksi = ", mprediksi)
        y = 0
        while(y < len(teta.columns)):
            mdteta[y] = perhitungandeltax(msigm,dataframe.iloc[i, 4],i,y)
            mtetab[y] = tetabaru(teta.iloc[0, y],alpha,mdteta[y])
            print (y)
            y = y+1
        while(y < len(teta.columns)):
            teta.iloc[0, y]= mtetab[y]
        dataframe.iloc[i, 4] = biasbaru(bias,alpha,mdbias)

def perhitunganfull(berapakali):
    for i in range(berapakali):
        perhitunganepoch(dataframe)

perhitunganfull(60)