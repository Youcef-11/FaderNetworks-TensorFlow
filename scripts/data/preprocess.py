#!/usr/bin/env python
## Youcef Chorfi

import os
import pathlib as Path
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import cv2
from readparams import getParams
from pathlib import Path
import random
import warnings
warnings.simplefilter("ignore")

params = getParams()
PATH_IMGS = params.get('PATH_IMGS')
ATTR_PATH = params.get('ATTR_PATH')
BATCH_SIZE = params.get('BATCH_SIZE')
IMG_SIZE = eval(params.get('IMG_SIZE'))
ONE_HOT = params.get('ONE_HOT')
SCALE_IMGS = params.get("SCALE_IMGS")
ATTR = params.get("ATTR")


"""
File : preprocess.py

Author : Youcef CHORFI

Data_loader : Class

Load all images path and return all images for the needed batch on demand

Reason to not fall in the out of memory peoblem when we try to load all images at the same time
"""

def denormalize(image):
    im = np.array(image)
    im = 127.5*(im + 1)
    return np.uint8(im)

class Data_loader :
    def __init__(self, params, split=None):
        self.__params = params
        self.__imgs_path = str(Path(__file__).parent)+'/'+params.get('PATH_IMGS')
        self.__attr_path = str(Path(__file__).parent)+'/'+params.get('ATTR_PATH')
        self.__batch_size = params.get('BATCH_SIZE')
        self.__img_size = eval(params.get('IMG_SIZE'))
        self.__attr_names = params.get("ATTR")
        self.__split = split
        self.__one_hot = params.get('ONE_HOT')
        self.__scale = params.get('SCALE_IMGS')
        if split :
            self.__data_path_train, self.__data_path_test = self.__load_imgs_path()
            self.__attr_train, self.__attr_test = self.__preprocess_attributes()
        else :
            self.__data_path_train = self.__load_imgs_path()
            self.__attr_train = self.__preprocess_attributes()



    def __load_imgs_path(self) :
        data_path = np.sort(np.array(os.listdir(self.__imgs_path)))
        self.__total_batch_number = len(data_path)//self.__batch_size
        data_path = data_path[:self.__total_batch_number*self.batch_size]
        data_path = data_path.reshape(self.__total_batch_number, self.__batch_size)

        if self.__split:
            batch_number = int(self.__total_batch_number*self.__split)
            data_path_train = data_path[:batch_number]
            data_path_test = data_path[batch_number:]
            return data_path_train, data_path_test
            
        return data_path

    def __preprocess_attributes(self):

        attr_lines = [line.rstrip() for line in open(self.__attr_path, 'r')]
        attr_names = np.array(attr_lines[1].split())
        if self.__attr_names :
            attr_indices = np.where(np.isin(attr_names,self.__attr_names))[0]
        else :
            attr_indices = np.arange(0,40)
        attributes = []
        if self.__one_hot :
            for i, line in enumerate(attr_lines[2:]):
                split = np.array(line.split()[1:])
                assert all(x in ['-1', '1'] for x in split[1:])
                attributes.append([[1,0] if x=='-1' else [0, 1] for x in split[attr_indices]])
            
            
            attributes = np.array(attributes[:self.__total_batch_number*self.__batch_size])
            attributes = attributes.reshape(self.__total_batch_number, self.__batch_size,len(attr_indices), 2)
        else :
            for i, line in enumerate(attr_lines[2:]):
                split = np.array(line.split()[1:])
                assert all(x in ['-1', '1'] for x in split[1:])
                attributes.append([0 if x=='-1' else 1 for x in split[attr_indices]])
            
            
            attributes = np.array(attributes[:self.__total_batch_number*self.__batch_size])
            attributes = attributes.reshape(self.__total_batch_number, self.__batch_size,len(attr_indices))
        
        if self.__split:
            batch_number = int(self.__total_batch_number*self.__split)
            attributes_train = attributes[:batch_number]
            attributes_test = attributes[batch_number:]
            return attributes_train, attributes_test

        return attributes


    def __iter__(self):
        for i in range(self.__data_path_train.shape[0]):
            batch_img = []
            for j in range(self.__batch_size) :
                image = mpimg.imread(self.__imgs_path+"/"+self.__data_path_train[i,j])
                image = cv2.resize(image, self.__img_size)
                if self.__scale: image = self.__normalize(image)
                batch_img.append(image)
            yield (np.array(batch_img), self.__attr_train[i])
    

    def __getitem__(self, i):
        batch_img = []
        for j in range(self.__batch_size) :
            image = mpimg.imread(self.__imgs_path+"/"+self.__data_path_train[i,j])
            image = cv2.resize(image, self.__img_size)
            if self.__scale: image = self.__normalize(image)
            batch_img.append(image)
        return (np.array(batch_img), self.__attr_train[i])
    
    def get_random_test_batch(self):
        batch_img = []
        rand = random.choice(range(len(self.__data_path_test)))
        for j in range(self.__batch_size) :
            image = mpimg.imread(self.__imgs_path+"/"+self.__data_path_test[rand,j])
            image = cv2.resize(image, self.__img_size)
            if self.__scale: image = self.__normalize(image)
            batch_img.append(image)
        return (np.array(batch_img), self.__attr_test[rand])
    
    def get_test_batches_iter(self):
        for i in range(self.__data_path_test.shape[0]):
            batch_img = []
            for j in range(self.__batch_size) :
                image = mpimg.imread(self.__imgs_path+"/"+self.__data_path_test[i,j])
                image = cv2.resize(image, self.__img_size)
                if self.__scale: image = self.__normalize(image)
                batch_img.append(image)
            yield (np.array(batch_img), self.__attr_test[i])
    
    @property
    def batch_size(self):
        return self.__batch_size
    @property
    def train_batch_number(self):
        return self.__data_path_train.shape[0]
    @property
    def test_batch_number(self):
        return self.__data_path_test.shape[0]

    def __normalize(self, image):    
        # Normalization between -1 et 1 
        return image/127.5 -1
         




if __name__=="__main__":
    # Data = Data_loader(params, split=0.9)
    # rand_img = random.choice(range(Data.batch_size))
    # batch_x, batch_y = Data.get_random_test_batch()
    # X, y = batch_x[rand_img], batch_y[rand_img]
    # X = denormalize(X)
    # print(y)
    # attr = params.get("ATTR")
    # plt.figure(1)
    # plt.imshow(X)
    # plt.title(f"Real : {attr[15]}:{y[15]}, {attr[20]}:{y[20]}")
    # plt.show()
    celebA = Data_loader(params, split=0.8)
    attr = params.get("ATTR")
    for i, batch in enumerate(celebA.get_test_batches_iter()) :
        X_test, y_test = batch
        print(f"test_batch n°{i}: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")
        plt.figure(1)
        plt.imshow(denormalize(X_test[12]))
        plt.title(f"{attr[3]}:{y_test[12][3]}, {attr[4]}:{y_test[12][4]}")
        plt.show()
    # for i, batch in enumerate(celebA) :
    #     X, y = batch
    #     os.system('clear')
    #     print(f"batch n°{i}: X.shape = {X.shape}, y.shape = {y.shape}")
    #     plt.figure(1)
    #     plt.imshow(X[12])
    #     plt.title(f"{attr[15]}:{y[12][15]}, {attr[20]}:{y[12][20]}")
    #     plt.show()
    #     if i == 100 : break
    # attr = params.get("ATTR")
    # X_train, y_train = celebA_train[12]
    # X_test, y_test = celebA_test[12]
    # print(f"Train set batch_number : {celebA_train.batch_number}")
    # print(f"Test set batch_number : {celebA_test.batch_number}")
    # plt.figure(1)
    # plt.imshow(X_test[12])
    # plt.title(f"{attr[0]}:{y_test[12][0]}, {attr[1]}:{y_test[12][1]}")
    # plt.figure(2)
    # plt.imshow(X_train[12])
    # plt.title(f"{attr[0]}:{y_train[12][0]}, {attr[1]}:{y_train[12][1]}")
    # plt.show()
