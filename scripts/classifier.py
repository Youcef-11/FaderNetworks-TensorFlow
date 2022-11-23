#!/usr/bin/env python
## Youcef Chorfi

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
from data import getParams, Data_loader
from Models import Classifier, attr_loss_accuracy
import tensorflow as tf
from tensorflow import keras
import numpy as np
from time import time
import random
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

params = getParams()

if __name__ == '__main__': 
    Data_train = Data_loader(params, split=0.2, train_test="train")
    Data_test = Data_loader(params, split=0.2, train_test="test")
    # Data = Data_loader(params)
    
    classifier = load_model(params.get("CLASSIFIER_PATH")+"/classifier", model_type = "c", params_name = 'params.npy', weights_name ='weights' , train = True)
    # classifier = Classifier(params)
    classifier.compile(optimizer= keras.optimizers.SGD())
    classifier.fit(Data_train=Data_train, Data_validation=Data_test)
    # classifier.evaluate(Data_test)
    # rand_batch = random.choice(range(Data.batch_number))
    # rand_img = random.choice(range(Data.batch_size))
    # batch_x, batch_y = Data[rand_batch]
    # X, y = batch_x[rand_img], batch_y[rand_img]
    # y =  tf.argmax(y, axis=-1).numpy()
    # print(y)
    # y_pred = tf.squeeze(tf.argmax(classifier.predict(tf.expand_dims(X,0)), axis=-1)).numpy()
    # print(y_pred)
    # print(accuracy_score(y,y_pred)*100)
    # attr = params.get("ATTR")
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(X)
    # plt.title(f"Real : {attr[15]}:{y[15]}, {attr[20]}:{y[20]}")
    # plt.subplot(1,2,2)
    # plt.imshow(X)
    # plt.title(f"Pred : {attr[15]}:{y_pred[15]}, {attr[20]}:{y_pred[15]}")
    # plt.show()





