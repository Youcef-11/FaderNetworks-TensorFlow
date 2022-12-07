#!/usr/bin/env python
## Gontran GILLES
## Youcef Chorfi **Update**

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
from data import getParams, Data_loader
import numpy as np
import random
from utils import *
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import accuracy_score
import cv2
import pandas as pd
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

params = getParams()


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])



if __name__ == '__main__':
    Data = Data_loader(params, split=0.9)
    
    fader, history = load_model_histroy(file=params.get('FADER_FILE'), train = False)

    fader.compile(
        ae_opt= tf.keras.optimizers.Adam(learning_rate=0.0002),
        dis_opt= tf.keras.optimizers.Adam(learning_rate=0.00002),
        ae_loss = tf.keras.losses.MeanSquaredError()
    )

    # hist = pd.DataFrame(history).plot()
    # plt.title("Fader results")
    # plt.xlabel("epochs")
    # plt.grid()
    # plt.show()

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(history['ae_loss'], label="Auto-encoder loss")
    plt.plot(history['ae_val_loss'], label="Auto-encoder validation loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid()
    plt.title("Auto-encoder losses")
    plt.subplot(2,2,2)
    plt.plot(history['dis_loss'], label="Discriminator loss")
    plt.plot(history['dis_val_loss'], label="Discriminator validation loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid()
    plt.title("Discriminator losses")
    plt.subplot(2,2,3)
    plt.plot(history['classifier_loss'], label="Classifier validation loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid()
    plt.title("Classifier validation loss")
    plt.subplot(2,2,4)
    plt.plot(history['dis_accuracy'], label="Discriminator F1-score")
    plt.plot(history['dis_val_accuracy'], label="Discriminator validation F1-score")
    plt.plot(history['classifier_acc'], label="Classifier validation F1-score")
    plt.xlabel("epochs")
    plt.legend()
    plt.grid()
    plt.title("F1-score")
    plt.show()
 

    batch = Data.get_random_test_batch()

    batch_x, batch_y = batch

    #rand_img = random.choice(range(Data.batch_size))
    # rand_imgs = random.sample(range(Data.batch_size),10)
    # X, y = [], []
    # for rand_img in rand_imgs:
    #     X.append(batch_x[rand_img])
    #     y.append(batch_y[rand_img])

    # X = np.array(X)
    # y = np.array(y)
    # y_perso = []
    # X_result = []
    # for i, att in enumerate(y):
    #     if att[0] == 0:
    #         y_perso.append(np.linspace(0.2,2,7))
    #     else:
    #         y_perso.append(np.linspace(0.8,-1,7))
    #     x = []
    #     x.append(cv2.cvtColor(denormalize(X[i]),cv2.COLOR_BGR2RGB))
    #     for attr in y_perso[-1]:
    #         X_recons = tf.squeeze(fader.ae.predict(tf.expand_dims(X[i],0),tf.expand_dims([attr],0)))
    #         x.append(cv2.cvtColor(denormalize(X_recons),cv2.COLOR_BGR2RGB))
    #     X_result.append(x)

    # im_tile = concat_tile(X_result)
    # cv2.imwrite('../results/results.jpg', im_tile)
    
    # plt.figure(figsize=(16,12))
    # for i, images in enumerate(X_result):
    #     for j, image in enumerate(images):
    #         plt.subplot(10,8,8*i+j+1)
    #         if 8*i+j == 0 : plt.title("Real")
    #         plt.imshow(image)
    #         plt.axis("off")
    
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()


    # fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
    # ax = [fig.add_subplot(2,2,i+1) for i in range(4)]

    # for a in ax:
    #     a.set_xticklabels([])
    #     a.set_yticklabels([])
    #     a.set_aspect('equal')

    # fig.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    # y_perso = deepcopy(y)
    # y_perso = np.where(0, y_perso==1, 1)
    # X_recons = tf.squeeze(fader.ae.predict(tf.expand_dims(X,0),tf.expand_dims(y_perso,0)))
    # X_recons = denormalize(X_recons)
    # X = denormalize(X)
    # attr = params.get("ATTR")
    # 
    # plt.subplot(1,2,1)
    # plt.imshow(X)
    # plt.axis("off")
    # plt.title(f"Real : {attr[0]}:{y[0]}")
    # plt.subplot(1,2,2)
    # plt.imshow(X_recons)
    # plt.axis("off")
    # plt.title(f"Fake : {attr[0]}:{y_perso[0]}")
    # plt.show()









