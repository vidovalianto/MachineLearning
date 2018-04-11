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

x1 = np.empty(150)
x2 = np.empty(150)
x3 = np.empty(150)
x4 = np.empty(150)
kelas = np.empty(150, dtype='S30')

dataframe = pd.read_csv("./data.csv")
dataframe = dataframe['teta'].str.split(',', expand=True)

fakta1 = np.empty(150, dtype='int32')
fakta2 = np.empty(150, dtype='int32')
fakta3 = np.empty(150, dtype='int32')

fakta1[0:50] = 1
fakta1[50:150] = 0
fakta2[0:50] = 0
fakta2[50:100] = 1
fakta2[100:150] = 0
fakta3[0:100] = 0
fakta3[100:150] = 1



teta1 = np.array([[0.1, 0.1, 0.1, 0.1], 
                       [0.1, 0.1, 0.1, 0.1],
                       [0.1, 0.1, 0.1, 0.1]])
bias1 = np.array([0.1,0.1,0.1])

teta2 = np.array([[0.1, 0.1, 0.1], 
                       [0.1, 0.1, 0.1],
                       [0.1, 0.1, 0.1]])
bias2 = np.array([0.1,0.1,0.1])

datatrain = dataframe



def sigm(hasilh):
    try:
        ans = (1/(1+math.exp( -hasilh )))
    except OverflowError:
        ans = float('inf')
    return ans



def back_prop(training,teta1,bias1,teta2,bias2):

    error1 = 0
    error2 = 0
    error3 = 0
    
    for i in range(training.iloc[:,0].size):
    
    
        h11 = sum(training.iloc[i,4:8]*teta1[0,0:4]) + bias1[0]
        h12 = sum(training.iloc[i,4:8]*teta1[1,0:4]) + bias1[1]
        h13 = sum(training.iloc[i,4:8]*teta1[2,0:4]) + bias1[2]
   
        s11 = sigm(h11)
        s12 = sigm(h12)
        s13 = sigm(h13)
        

        dbias11 = 2*(s11 - training.iloc[i,0]) * (1-s11)*s11
        dw11 = dbias11 * training.iloc[i,4]
        dw14 = dbias11 * training.iloc[i,5]
        dw17 = dbias11 * training.iloc[i,6]
        dw110 = dbias11 * training.iloc[i,7]


        dbias12 = 2*(s12 - training.iloc[i,0]) * (1-s12)*s12
        dw12 = dbias12 * training.iloc[i,4]
        dw15 = dbias12 * training.iloc[i,5]
        dw18 = dbias12 * training.iloc[i,6]
        dw111 = dbias12 * training.iloc[i,7]

        dbias13 = 2*(s13 - training.iloc[i,0]) * (1-s13)*s13
        dw13 = dbias13 * training.iloc[i,4]
        dw16 = dbias13 * training.iloc[i,5]
        dw19 = dbias13 * training.iloc[i,6]
        dw112 = dbias13 * training.iloc[i,7]

        dw1 = np.array([[dw11, dw14, dw17, dw110],
                        [dw12, dw15, dw18, dw111],
                        [dw13, dw16, dw19, dw112]])
        
        dbias1 = np.array([dbias11, dbias12, dbias13])

        teta1 = teta1 - alpha * dw1
        bias1 = bias1 - alpha * dbias1

        h21 = s11 * teta2[0][0] + s12 * teta2[0][1] + s13 * teta2[0][2] + bias2[0]
        h22 = s11 * teta2[1][0] + s12 * teta2[1][1] + s13 * teta2[1][2] + bias2[1]
        h23 = s11 * teta2[2][0] + s12 * teta2[2][1] + s13 * teta2[2][2] + bias2[2]
    
        s21 = sigm(h21)
        s22 = sigm(h22)
        s23 = sigm(h23)

    
        error1 = error1 + (s21 - training.iloc[i,0])**2
        error2 = error2 + (s22 - training.iloc[i,1])**2
        error3 = error3 + (s23 - training.iloc[i,2])**2

    
        tau21 = 2*(s21 - training.iloc[i,0]) * (1-s21)*s21
        dw21 = tau21 * s11
        dw24 = tau21 * s12
        dw27 = tau21 * s13
 
    
        tau22 = 2*(s22 - training.iloc[i,1]) * (1-s22)*s22
        dw22 = tau22 * s11
        dw25 = tau22 * s12
        dw28 = tau22 * s13

        
        tau23 = 2*(s23 - training.iloc[i,2]) * (1-s23) * s23
        dw23 = tau23 * s11
        dw26 = tau23 * s12
        dw29 = tau23 * s13

       
        dw2 = np.array([[dw21, dw24, dw27],
                        [dw22, dw25, dw28],
                        [dw23, dw26, dw29]])
    
        dbias2 = np.array([tau21, tau22, tau23])

        
        
        teta2 = teta2 - alpha * dw2
        bias2 = bias2 - alpha * dbias2
        
    error = np.array([error1, error2,error3])
    error = error/(training.iloc[:,0].size)
    error = np.mean(error)
    
    return np.array([[error],[teta1, bias1, teta2, bias2]])

def prediksi(sigm):
    if sigm < 0.5:
        return 0
    else:
        return 1
    
def perhitunganakurasi(data1,data2):
    dataakurasi = np.zeros(data1.iloc[:,0].size, dtype=bool)
    #print (data1)
    #print (data2)
    #print( checking
    for i in range(data1.iloc[:,0].size):
        dataakurasi[i] = np.array_equal(data1.iloc[i].values, data2.iloc[i].values)
    
    return dataakurasi

def test(testing, params):
    theta1 = np.array(params[0])
    bias1 = np.array(params[1])
    theta2 = np.array(params[2])
    bias2 = np.array(params[3])
    
    prediksi1 = np.zeros(testing.iloc[:,0].size, dtype='int32')
    prediksi2 = np.zeros(testing.iloc[:,0].size, dtype='int32')
    prediksi3 = np.zeros(testing.iloc[:,0].size, dtype='int32')

    error1 = 0
    error2 = 0
    error3 = 0
    
    
    for i in range(testing.iloc[:,0].size):
        h11 = sum(testing.iloc[i,4:8]*theta1[0,0:4]) + bias1[0]
        h12 = sum(testing.iloc[i,4:8]*theta1[1,0:4]) + bias1[1]
        h13 = sum(testing.iloc[i,4:8]*theta1[2,0:4]) + bias1[2]
 
        s11 = sigm(h11)
        s12 = sigm(h12)
        s13 = sigm(h13)
  
    
        h21 = s11 * theta2[0][0] + s12 * theta2[0][1] + s13 * theta2[0][2] + bias2[0]
        h22 = s11 * theta2[1][0] + s12 * theta2[1][1] + s13 * theta2[1][2] + bias2[1]
        h23 = s11 * theta2[2][0] + s12 * theta2[2][1] + s13 * theta2[2][2] + bias2[2]

    
        s21 = sigm(h21)
        s22 = sigm(h22)
        s23 = sigm(h23)

        
        prediksi1[i] = prediksi(s21)
        prediksi2[i] = prediksi(s22)
        prediksi3[i] = prediksi(s23)
    
        error1 = error1 + (s21 - testing.iloc[i,0])**2
        error2 = error2 + (s22 - testing.iloc[i,1])**2
        error3 = error3 + (s23 - testing.iloc[i,2])**2
       
        
    error = np.array([error1, error2,error3])
    error = error/(testing.iloc[:,0].size)
    error = np.mean(error)

    predict_table = pd.DataFrame({'prediksi 1':prediksi1,
                                  'prediksi 2':prediksi2,
                                  'prediksi 3':prediksi3})
    
  
    conditional = perhitunganakurasi(testing.iloc[:,0:3], predict_table)

    unique, count = np.unique(conditional, return_counts=True)
   
    c = np.where(unique == True)
    if(c[0].size != 0):   
        akurasi = (count[c[0][0]] / (testing.iloc[:,0].size)) * 100
    else:
        akurasi = 0
    
    return np.array([error,akurasi])


df = pd.DataFrame({'x1':x1,
                   'x2':x2,
                   'x3':x3,
                   'x4':x4,
                   'kelas':kelas,
                   'fakta 1':fakta1,
                   'fakta 2':fakta2,
                   'fakta 3':fakta3})



error_training_epoch = np.empty(100)
error_testing_epoch = np.empty(100)
akurasi_epoch = np.empty(100)

error_training = 0
error_testing = 0
akurasi = 0


testing = df.iloc[0:30]
training = df.iloc[30:150]



training_result = back_prop(training, teta1, bias1, teta2, bias2)
error_training = error_training + training_result[0][0]

testing_result = test(testing,training_result[1])
error_testing = error_testing + testing_result[0]
akurasi = akurasi + testing_result[1]


for i in range(100):
    testing = df.iloc[30:60]
    training = df.iloc[0:30]
    training = training.append(df.iloc[60:150])

    
    training_result = back_prop(training,training_result[1][0],training_result[1][1],training_result[1][2],training_result[1][3])
    error_training = error_training + training_result[0][0]

    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    testing = df.iloc[60:90]
    training = df.iloc[0:60]
    training = training.append(df.iloc[90:150])
    
    training_result = back_prop(training,training_result[1][0],training_result[1][1],training_result[1][2],training_result[1][3])
    error_training = error_training + training_result[0][0]
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]


    testing = df.iloc[90:120]
    training = df.iloc[0:90]
    training = training.append(df.iloc[120:150])


    
    training_result = back_prop(training,training_result[1][0],training_result[1][1],training_result[1][2],training_result[1][3])
    error_training = error_training + training_result[0][0]

    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]


    testing = df.iloc[120:150]
    training = df.iloc[0:120]

  
    training_result = back_prop(training,training_result[1][0],training_result[1][1],training_result[1][2],training_result[1][3])
    error_training = error_training + training_result[0][0]

    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]



    mean_error_training = error_training/5
    mean_error_testing = error_testing/5
    mean_akurasi = akurasi/5
    error_training_epoch[i] = mean_error_training
    error_testing_epoch[i] = mean_error_testing
    akurasi_epoch[i] = mean_akurasi
    
    if(i!=30):
        error_training = 0
        error_testing = 0
        akurasi = 0

        testing = df.iloc[0:30]
        training = df.iloc[30:150]
      
        training_result = back_prop(training,training_result[1][0],training_result[1][1],training_result[1][2],training_result[1][3])
        error_training = error_training + training_result[0][0]
 
        testing_result = test(testing,training_result[1])
        error_testing = error_testing + testing_result[0]
        akurasi = akurasi + testing_result[1]



print(error_training_epoch)
print(error_testing_epoch)
print(akurasi_epoch)


plt.plot(error_training_epoch,'r',error_testing_epoch,'g')
plt.show()
