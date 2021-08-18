#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:10:03 2021

@author: zeeshan
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";


import keras
from keras import layers
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


df = pd.read_csv('/home/zeeshan/AD/DEG_AfterCorrRemoval.csv',delimiter = ",")
# df = pd.read_csv('/home/zeeshan/AD/DMP_UNLABELLED.csv',delimiter = ",", dtype={'SampleID': str})
X_train = df.iloc[:,:]
X_train.shape

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax.shape


# encoding_dim = 32
input_img = keras.Input(shape=(34375,)) #DMP:478245 / DEG: 34375
#######

encoded = layers.Dense(256, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(2, activation='relu')(encoded)

decoded = layers.Dense(16, activation='relu')(encoded)
decoded = layers.Dense(32, activation='relu')(decoded)
decoded = layers.Dense(64, activation='relu')(decoded)
decoded = layers.Dense(256, activation='relu')(decoded)
decoded = layers.Dense(34375, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse') #binary_crossentropy

#####

encoder = keras.Model(input_img, encoded)


x_train, x_test = train_test_split(X_train_minmax, test_size=0.2)


autoencoder.fit(x_train, x_train,
                epochs=150,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
x_cord = list(range(20))
y1 = x_test[4][2000:2020]
y2 = decoded_imgs[4][2000:2020]
plt.plot(x_cord,y1) 
plt.plot(x_cord,y2) 

# Encoding original data using the trained encoder
new_data = encoder.predict(X_train)
new_data_df = pd.DataFrame(new_data)
new_data_df.to_csv('/home/zeeshan/AD/Encoded_Data/DEG_encoded_2.csv', header=True, index=False, sep=',')


