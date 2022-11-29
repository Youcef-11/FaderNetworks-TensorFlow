#!/usr/bin/env python
## Youcef Chorfi

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
import pandas as pd
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

params = getParams()

if __name__ == '__main__':
    Data = Data_loader(params, split=0.9)
    
    fader, history = load_model_histroy(file=params.get('FADER_FILE'), train = False)

    fader.compile(
        ae_opt= tf.keras.optimizers.Adam(learning_rate=0.0002),
        dis_opt= tf.keras.optimizers.Adam(learning_rate=0.00002),
        ae_loss = tf.keras.losses.MeanSquaredError()
    )

    hist = pd.DataFrame(history).plot()
    plt.grid()
    plt.show()

    for i, batch in enumerate(Data.get_test_batches_iter()) :
        print(i,"*"*40)
        batch_x, batch_y = batch
        rand_img = random.choice(range(Data.batch_size))
        X, y = batch_x[rand_img], batch_y[rand_img]
        y_perso = deepcopy(y)
        y_perso = 1 - y
        X_recons = tf.squeeze(fader.ae.predict(tf.expand_dims(X,0),tf.expand_dims(y_perso,0)))
        X_recons = denormalize(X_recons)
        X = denormalize(X)
        attr = params.get("ATTR")
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(X)
        plt.title(f"Real : {attr[0]}:{y[0]}")
        plt.subplot(1,2,2)
        plt.imshow(X_recons)
        plt.title(f"Fake : {attr[0]}:{y_perso[0]}")
        plt.show()





