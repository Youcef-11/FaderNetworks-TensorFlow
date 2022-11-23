#!/usr/bin/env python
## Youcef Chorfi

import os
import pathlib as Path
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from readparams import getParams
from pathlib import Path
from tqdm import tqdm
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

class Data_loader :
    def __init__(self, params, train_test="train", split=None):
        self.__params = params
        self.__imgs_path = str(Path(__file__).parent)+'/'+params.get('PATH_IMGS')
        self.__attr_path = str(Path(__file__).parent)+'/'+params.get('ATTR_PATH')
        self.__batch_size = params.get('BATCH_SIZE')
        self.__img_size = eval(params.get('IMG_SIZE'))
        self.__attr = params.get("ATTR")
        self.__split = split
        self.__train_test = train_test
        # self.__one_hot = one_hot
        self.__scale = params.get('SCALE_IMGS')
        self.__data_path = self.__load_imgs_path()
        self.__attr = self.__preprocess_attributes()



    def __load_imgs_path(self) :
        data_path = np.sort(np.array(os.listdir(self.__imgs_path)))
        self.__batch_total_number = len(data_path)//self.__batch_size
        self.__batch_number = self.__batch_total_number
        data_path = data_path[:self.__batch_number*self.batch_size]
        data_path = data_path.reshape(self.__batch_number, self.__batch_size)
        if self.__split:
            self.__batch_number = int(self.__batch_total_number*self.__split)
            if self.__train_test == "train":
                data_path = data_path[:self.__batch_number]
            elif self.__train_test == "test":
                data_path = data_path[self.__batch_number:]
            else:
                raise("You shoud precise train or test if you want Data for train or test")
        return data_path


    def __iter__(self):
        for i in range(self.__data_path.shape[0]):
            batch_img = []
            for j in range(self.__batch_size) :
                image = mpimg.imread(self.__imgs_path+"/"+self.__data_path[i,j])
                image = cv2.resize(image, self.__img_size)
                if self.__scale: image = image/255
                batch_img.append(image)
            yield (np.array(batch_img), self.__attr[i])
    

    def __getitem__(self, i):
        batch_img = []
        for j in range(self.__batch_size) :
            image = mpimg.imread(self.__imgs_path+"/"+self.__data_path[i,j])
            image = cv2.resize(image, self.__img_size)
            if self.__scale: image = image/255
            batch_img.append(image)
        return (np.array(batch_img), self.__attr[i])
    
    @property
    def batch_size(self):
        return self.__batch_size
    @property
    def batch_number(self):
        return self.__data_path.shape[0]

    def __preprocess_attributes(self):

        attr_lines = [line.rstrip() for line in open(self.__attr_path, 'r')]
        attr_names = np.array(attr_lines[1].split())
        if self.__attr :
            attr_indices = np.where(np.isin(attr_names,self.__attr))[0]
        else :
            attr_indices = np.arange(0,40)
        attributes = []
        for i, line in enumerate(attr_lines[2:]):
            image_id = i + 1
            split = np.array(line.split()[1:])
            assert all(x in ['-1', '1'] for x in split[1:])
            attributes.append([[1,0] if x=='-1' else [0, 1] for x in split[attr_indices]])
        
        
        attributes = np.array(attributes[:self.__batch_total_number*self.__batch_size])
        attributes = attributes.reshape(self.__batch_total_number, self.__batch_size,len(attr_indices), 2)
        if self.__split:
            if self.__train_test == "train":
                attributes = attributes[:self.__batch_number]
            elif self.__train_test == "test":
                attributes = attributes[self.__batch_number:]
            else:
                raise("You shoud precise train or test if you want Data for train or test")
        # if self.__one_hot :
        #     attributes = attributes.reshape(self.__batch_number, self.__batch_size,len(attr_indices),1,1)
        #     attributes = np.broadcast_to(attributes, (self.__batch_number, self.__batch_size,len(attr_indices),2,2))

        return attributes
         




if __name__=="__main__":
    celebA_train = Data_loader(params, split=0.8, train_test="train")
    celebA_test = Data_loader(params, split=0.8, train_test="test")
    for i, batch in enumerate(celebA_train) :
        X, y = batch
        attr = params.get("ATTR")
        os.system('clear')
        print(f"batch n°{i}: X.shape = {X.shape}, y.shape = {y.shape}")
        plt.figure(1)
        plt.imshow(X[12])
        plt.title(f"{attr[15]}:{y[12][15]}, {attr[20]}:{y[12][20]}")
        plt.show()
        if i == 100 : break
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
