#!/usr/bin/env python
## Youcef Chorfi

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
from data import getParams, Data_loader
from Models import Classifier
import numpy as np
import random
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

params = getParams()

if __name__ == '__main__':
    Data = Data_loader(params, split=0.9)
    
    classifier, history = load_model_histroy(file="classifier_1669469208", train = True)
    
    # classifier = Classifier(params)
    classifier.compile(optimizer= tf.keras.optimizers.RMSprop())
    # classifier.fit(Data)

    plt.figure()
    hist = pd.DataFrame(history).plot()
    plt.show()

    for i, batch in enumerate(Data.get_test_batches_iter()) :
        print(i,"*"*40)
        batch_x, batch_y = batch
        rand_img = random.choice(range(Data.batch_size))
        X, y = batch_x[rand_img], batch_y[rand_img]
        X = denormalize(X)
        print(y)
        y_pred = tf.squeeze(tf.round(classifier.predict(tf.expand_dims(X,0)))).numpy()
        print(y_pred)
        print(accuracy_score(y,y_pred)*100)
        attr = params.get("ATTR")
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(X)
        plt.title(f"Real : {attr[3]}:{y[3]}, {attr[4]}:{y[4]}")
        plt.subplot(1,2,2)
        plt.imshow(X)
        plt.title(f"Pred : {attr[3]}:{y_pred[3]}, {attr[4]}:{y_pred[4]}")
        plt.show()





