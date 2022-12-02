#!/usr/bin/env python

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
from data import getParams, Data_loader
from Models import Classifier, Fader, AutoEncoder, save_model_weights
import numpy as np
import random
from utils import *
import matplotlib.pyplot as plt
from termcolor import colored


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)



from time import time
from utils import *


params = getParams()

if __name__ == "__main__":

    Data = Data_loader(params, split=0.9)

    if params.get('FADER_FILE'):
        f, history = load_model_histroy(file=params.get('FADER_FILE'), train = True)
        best_val_loss = history['ae_val_loss'][-1]
        best_val_acc = history['classifier_acc'][-1]
    else:
        f = Fader(params)
        history = {'ae_loss':[], 'dis_loss':[], 'dis_accuracy':[], 
                'ae_val_loss':[], 'dis_val_loss':[], 'dis_val_accuracy':[], 
                'classifier_loss':[], 'classifier_acc':[]}

        best_val_loss = np.inf
        best_val_acc = 0


    if params.get('CLASSIFIER_FILE'):
        classifier, _ = load_model_histroy(file=params.get('CLASSIFIER_FILE'), train = False)
        classifier.compile(optimizer= tf.keras.optimizers.RMSprop())

    f.compile(
        ae_opt= tf.keras.optimizers.Adam(learning_rate=2e-4),
        dis_opt= tf.keras.optimizers.Adam(learning_rate=2e-5),
        ae_loss = tf.keras.losses.MeanSquaredError()
    )

    random.seed(10)
    train_batch_number = 1000
    batchs_train = random.sample(range(Data.train_batch_number),train_batch_number)
    batch_test_number = 50
    n_epoch = params.get("N_EPOCH")
    lambda_dis = 0
    n_iter = 0

    for epoch in range(n_epoch):
        print(colored("Training Model...","blue"))
        #Training
        ae_loss_tab = []
        dis_loss_tab = []
        dis_accuracy_tab = []

        for step, batch in enumerate(batchs_train) :
            t = time()
            ae_loss, dis_loss,  dis_acc = f.train_step(Data[batch], 0.003)

            ae_loss_tab.append(ae_loss)
            dis_loss_tab.append(dis_loss)
            dis_accuracy_tab.append(dis_acc)
            if step < train_batch_number-1:
                print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{train_batch_number} : ", colored(f"dis_accuracy = {dis_acc.numpy():.3f}, ","green"), colored(f"ae_loss : {ae_loss:.3f},", "red"), colored(f"dis_loss : {dis_loss:.3f},", "red"), colored(f"lambda_dis : {lambda_dis:.6f},", "magenta") ,"calculé en :", colored(f"{time() - t:.3f}s", "yellow"), end='\r')
            else:
                print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{train_batch_number} : ", colored(f"dis_accuracy = {dis_acc.numpy():.3f}, ","green"), colored(f"ae_loss : {ae_loss:.3f},", "red"), colored(f"dis_loss : {dis_loss:.3f},", "red"), colored(f"lambda_dis : {lambda_dis:.6f},", "magenta") ,"calculé en :", colored(f"{time() - t:.3f}s", "yellow"))
            n_iter += 1
            lambda_dis = 0.002*min(n_iter/320, 1)
            
        history['ae_loss'].append(np.mean(ae_loss_tab))
        history['dis_loss'].append(np.mean(dis_loss_tab))
        history['dis_accuracy'].append(np.mean(dis_accuracy_tab))

        # Validation
        recon_val_loss = []
        dis_val_loss = []
        dis_val_accuracy = []
        clf_loss = []
        clf_acc = []

        print(colored("Evaluating Model...","blue"))
        for step in range(batch_test_number):
            t = time()
            batch = Data.get_random_test_batch()
            ae_loss, dis_loss, dis_acc = f.evaluate_on_val(batch, lambda_dis)

            if params.get('CLASSIFIER_FILE'):
                clf_l, clf_a  = classifier.eval_fader_on_batch(batch, f)
                clf_loss.append(clf_l)
                clf_acc.append(clf_a)

            recon_val_loss.append(ae_loss)
            dis_val_loss.append(dis_loss)
            dis_val_accuracy.append(dis_acc)

            if step < batch_test_number-1:
                print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{batch_test_number} : ", colored(f"dis_accuracy = {dis_acc.numpy():.3f}, ","green"), colored(f"classifier_accuracy = {clf_a.numpy():.4}, ","green"), colored(f"ae_loss : {ae_loss:.3f}", "red"), colored(f"dis_loss : {dis_loss:.3f}", "red") ,"calculé en :", colored(f"{time() - t:.3f}s", "yellow"),end='\r')
            else:
                print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{batch_test_number} : ", colored(f"dis_accuracy = {dis_acc.numpy():.3f}, ","green"), colored(f"classifier_accuracy = {clf_a.numpy():.4}, ","green"), colored(f"ae_loss : {ae_loss:.3f}", "red"), colored(f"dis_loss : {dis_loss:.3f}", "red") ,"calculé en :", colored(f"{time() - t:.3f}s", "yellow"))

        history['ae_val_loss'].append(np.mean(recon_val_loss))
        history['dis_val_loss'].append(np.mean(dis_val_loss))
        history['dis_val_accuracy'].append(np.mean(dis_val_accuracy))
        history['classifier_loss'].append(np.mean(clf_loss))
        history['classifier_acc'].append(np.mean(clf_acc))



        # Save the best model at each time
        # We have 2 criteria for saving the model, the one that reconstructs the best (smallest reconstruction loss)
        # And the one whose trained classifier recognizes the attributes used to reconstruct the image
        if history['ae_val_loss'][-1] < best_val_loss:
            # En réalité il suffit de sauvegarder l'autoencoder, le discriminator ne servant a rien pour l'inférance
            best_val_loss = history['ae_val_loss'][-1]
            save_model_weights(model=f, h=history, file='fader_male_loss', acc=history['classifier_acc'][-1])

        elif params.get('CLASSIFIER_FILE') and history['classifier_acc'][-1] > best_val_acc:
            best_val_acc = history['classifier_acc'][-1]
            save_model_weights(model=f, h=history, file='fader_male_class', acc=best_val_acc)
        else :
            save_model_weights(model=f, h=history, file='fader_male', acc=history['classifier_acc'][-1])