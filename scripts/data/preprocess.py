#!/usr/bin/env python
## Youcef Chorfi

import os
import pathlib as Path
import numpy as np
import matplotlib.image as mpimg
import cv2
import tensorflow as tf
from read_params import Read_yaml

params = Read_yaml()


"""
File : preprocess.py

Author : Youcef CHORFI

Data_loader : Class

Load all images path and return all images for the needed batch on demand

Reason to not fall in the out of memory peoblem when we try to load all images at the same time
"""

class Data_loader :
    def __init__(self):
        self.__imgs_path = os.path.abspath(params['PATH_IMGS'])
        self.__attr_path = os.path.abspath(params['ATTR_PATH'])
        self.__batch_size = params['BATCH_SIZE']
        self.__img_size = eval(params['IMG_SIZE'])
        self.__one_hot = params['ONE_HOT']
        self.__data_path = self.__load_imgs_path()
        self.__attr = self.__preprocess_attributes()



    def __load_imgs_path(self) :
        data_path = np.array(os.listdir(self.__imgs_path))
        self.__batch_number = len(data_path)//self.__batch_size
        data_path = data_path[:self.__batch_number*self.__batch_size]
        return data_path.reshape(self.__batch_number, self.__batch_size)


    def __iter__(self):
        for i in range(self.__batch_number):
            batch_img = []
            for j in range(self.__batch_size) :
                image = mpimg.imread(self.__imgs_path+"/"+self.__data_path[i,j])
                image = cv2.resize(image, self.__img_size)
                batch_img.append(image)
            yield (np.array(batch_img), self.__attr[i])
    

    def __getitem__(self, i):
        batch_img = []
        for j in range(self.__batch_size) :
            image = mpimg.imread(self.__imgs_path+"/"+self.__data_path[i,j])
            image = cv2.resize(image, self.__img_size)
            batch_img.append(image)
        return (np.array(batch_img), self.__attr[i])
    
    @property
    def batch_size(self):
        return self.__batch_size

    def __preprocess_attributes(self):

        attr_lines = [line.rstrip() for line in open(self.__attr_path, 'r')]

        attributes = []
        for i, line in enumerate(attr_lines[2:]):
            image_id = i + 1
            split = line.split()
            assert len(split) == 41
            assert split[0] == ('%06i.jpg' % image_id)
            assert all(x in ['-1', '1'] for x in split[1:])
            A=[]
            for j, value in enumerate(split[1:]):
                A.append(int(value))
            attributes.append(A)
        
        attributes = np.array(attributes[:self.__batch_number*self.__batch_size])
        attributes = attributes.reshape(self.__batch_number, self.__batch_size,40)

        return tf.one_hot(attributes, depth=2, axis=-1) if self.__one_hot else attributes
         




if __name__=="__main__":
    celebA = Data_loader()
    for i,j in enumerate(celebA) :
        print(f"batch n°{i}: X.shape = {j[0].shape}, y.shape = {j[1].shape}")
        if i == 100 : break
    print(f"batch n°1500: X.shape = {celebA[1500][0].shape}, y.shape = {celebA[1500][1].shape}")
