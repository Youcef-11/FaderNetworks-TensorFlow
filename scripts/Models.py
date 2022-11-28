#!/usr/bin/env python
## Youcef Chorfi

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from termcolor import colored
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# Time for tensor board
from time import time

from tensorflow.keras import Model
# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Reshape, ReLU, Activation,
                                    LeakyReLU, Dropout, BatchNormalization, Conv2DTranspose)




from data import getParams, Data_loader
import warnings
warnings.simplefilter("ignore")

params = getParams()
IMG_SIZE = eval(params.get('IMG_SIZE'))



@tf.function
def attr_loss(y, y_hat, used_loss=tf.keras.losses.BinaryCrossentropy()):
    """
    Compute the macro soft F1-score and the cost function across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
        f1_score (scalar Tensor): value of the f1_score for the batch
    """
    # y = tf.cast(y, tf.float32)
    # y_hat = tf.cast(y_hat, tf.float32)
    y_hat = tf.reshape(y_hat,y.shape)
    cost=0
    accuracy=0
    # tp = tf.reduce_sum(tf.round(y_hat) * y, axis=0)
    # fp = tf.reduce_sum(tf.round(y_hat) * (1 - y), axis=0)
    # fn = tf.reduce_sum(tf.round(1 - y_hat) * y, axis=0)
    # f1_score = 2*tp / (2*tp + fn + fp + 1e-16)
    for i in range(y.shape[1]):
      cost += used_loss(y[:,i],y_hat[:,i])
    return cost



def save_model_weights(model, h, file, acc):
    '''
    Save model weights, params and history
    Args:
    model : tf model
    file : str, classifier or fader or ae
    '''
    print(colored("Saving Network Weights and History...","magenta"))
    t = str(100*acc)
    path = str(Path(__file__).parent)+'/'+params.get("MODELS_PATH")
    model.save_weights(path+'/'+file+'_'+t+'/'+'weights')
    np.save(path+'/'+file+'_'+t+'/'+'params', model.params)
    np.save(path+'/'+file+'_'+t+'/'+'history', h)


def enc_dec_model(params):
    encoder=Sequential([Conv2D(16, kernel_size=4, strides=2, padding="same", input_shape=(*IMG_SIZE,3)),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2),
                           Conv2D(32, kernel_size=4, strides=2, padding="same"),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2),
                           Conv2D(64, kernel_size=4, strides=2, padding="same"),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2),
                           Conv2D(128, kernel_size=4, strides=2, padding="same"),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2),
                           Conv2D(256, kernel_size=4, strides=2, padding="same"),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2),
                           Conv2D(512, kernel_size=4, strides=2, padding="same"),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2),
                           Conv2D(512, kernel_size=4, strides=2, padding="same"),
                           BatchNormalization(),
                           LeakyReLU(alpha=0.2)
                            ], name = "encoder")

    dec_input_shape = encoder.layers[-1].output_shape[-3:-1]
    dec_input_shape = (*dec_input_shape, 512+len(params.get("ATTR")))

    decoder=Sequential([Conv2DTranspose(512, kernel_size=4, strides=2, padding="same", input_shape=dec_input_shape),
                        BatchNormalization(),
                        ReLU(),
                        Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
                        BatchNormalization(),
                        ReLU(),
                        Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                        BatchNormalization(),
                        ReLU(),
                        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                        BatchNormalization(),
                        ReLU(),
                        Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
                        BatchNormalization(),
                        ReLU(),
                        Conv2DTranspose(32, kernel_size=4, strides=2, padding="same"),
                        BatchNormalization(),
                        ReLU(),
                        Conv2DTranspose(3, kernel_size=4, strides=2, padding="same"),
                        BatchNormalization(),
                        Activation('tanh')
                        ], name = "decoder")

    return encoder, decoder


def discriminator(params):
    discriminator = Sequential([
                                Conv2D(512, 4, 2, 'same', input_shape = (2,2,512)),
                                BatchNormalization(),
                                LeakyReLU(0.2),
                                Dropout(0.3),
                                Flatten(),
                                Dense(512, activation=LeakyReLU(0.2)),
                                Dense(len(params.get("ATTR")), activation='sigmoid')
                               ],
                               name = "discriminator")
    return discriminator


class Classifier(Model):
    '''
    Multi-label Classification
    Input : Images of shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    Output : y_preds (float32 Tensor) probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
    '''
    def __init__(self, params, history=None):
        super(Classifier, self).__init__()
        self.params = params
        if history:
            self.history = history
            self.best_metrics = self.history['val_loss'][-1]
        else:
            self.history = {'loss' : [], 'metrics': [], 'val_loss' : [], 'val_metrics' : []}
            self.best_metrics = 0

        self.model, _ = enc_dec_model(params)
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Dense(len(params.get("ATTR")), activation='sigmoid'))
        # self.build((None, 256,256,3))

    def get_optimizers(self):
        return (self.opt,)



    def compile(self, optimizer, loss=attr_loss, metrics=tf.keras.metrics.BinaryAccuracy()):
        super(Classifier, self).compile()
        self.opt = optimizer
        self.loss  = loss
        self.metrics = metrics


    @tf.function
    def eval_on_batch(self, data):
        x,y = data

        self.model.trainable = False
        y_preds = self.model(x)
        loss = self.loss(y, y_preds)
        metrics = self.metrics(y, y_preds)
        return loss, metrics


    @tf.function
    def train_step(self, data):
        x,y = data
        self.model.trainable = True
        with tf.GradientTape() as tape:
            y_preds=self(x)
            loss = self.loss(y, y_preds)
            metrics = self.metrics(y, y_preds)
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        return loss, metrics
    
    @tf.function
    def eval_fader_on_batch(self, data, fader):
        """
        This function evaluates the performance of the on images reconstructed with "random" attributes
        """
        x,y = data
        fader.trainable = False
        self.trainable = False

        assert np.isin(fader.params.get("ATTR"), self.params.get("ATTR")).all()

        attr_arg = np.where(np.isin(np.array(self.params.get("ATTR")), np.array(fader.params.get("ATTR"))))[0]
        
        x_ae = fader.ae.predict(x, y = 1-y)
        # Reconstuction
        clf_preds = self.model(x_ae)
        clf_preds = [clf_preds[:, i] for i in attr_arg]
        fader.trainable = True
        return self.loss(1-y, clf_preds), self.metrics(1-y, clf_preds)
    
    def fit(self, Data) :

        n_epoch = params.get("N_EPOCH")
        epoch_stop = 0
        for epoch in range(n_epoch):
            print(colored("Training Model...","blue"))
            loss = []
            metrics = []
            for step, batch in enumerate(Data) :
                t = time()
                l,a = self.train_step(batch)

                loss.append(l)
                metrics.append(a)
                if step < Data.train_batch_number-1:
                    print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{Data.train_batch_number} : ", colored(f"metrics = {a.numpy():.3f}, ","green"), colored(f"loss = {l.numpy():.3f}", "red"), "calculé en :", colored(f"{time() - t:.3f}s", "yellow"), end="\r")
                else:
                    print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{Data.train_batch_number} : ", colored(f"metrics = {a.numpy():.3f}, ","green"), colored(f"loss = {l.numpy():.3f}", "red"), "calculé en :", colored(f"{time() - t:.3f}s", "yellow"))
            self.history['loss'].append(np.mean(loss))
            self.history['metrics'].append(np.mean(metrics))
            
            print(colored("Evaluating Model...","blue"))
            for step, batch in enumerate(Data.get_test_batches_iter()):
                #Eval loop 
                loss = []
                metrics = []
                t = time()
                #l, a= self.eval_on_batch(batch)
                l, a= self.eval_on_batch(Data.get_random_test_batch())
                loss.append(l)
                metrics.append(a)
                if step < Data.test_batch_number-1:
                    print(f"validation batch : {1 + step}/{Data.test_batch_number} : ", colored(f"val_metrics = {a.numpy():.3f}, ","green"), colored(f"val_loss = {l.numpy():.3f}", "red"), "calculé en :", colored(f"{time() - t:.3f}s", "yellow"), end="\r")
                else:
                    print(f"validation batch : {1 + step}/{Data.test_batch_number} : ", colored(f"val_metrics = {a.numpy():.3f}, ","green"), colored(f"val_loss = {l.numpy():.3f}", "red"), "calculé en :", colored(f"{time() - t:.3f}s", "yellow"))

            print("Evaluation results :", colored(f"val_metrics = {np.mean(metrics):.3f}, ","green"), colored(f"val_loss = {np.mean(loss):.3f}", "red"))
            
            # Peut nous permettre de tracer un graph.
            self.history['val_loss'].append(np.mean(loss))
            self.history['val_metrics'].append(np.mean(metrics))

            if self.history['val_metrics'][-1] > self.best_metrics:
                epoch_stop = epoch 
                self.best_metrics = self.history['val_metrics'][-1]
                save_model_weights(model=self, h=self.history, file='classifier', acc=self.best_metrics)
            
            if abs(epoch-epoch_stop+1)>=10:
                print(colored("Training finished...","blue"))
                break
            

    def call(self, x):
        x = self.model(x)
        return x

class AutoEncoder(Model):
    """
    The presence of this class is due to the fact that the decoder needs 
    the latent representation z, and the attributes y to reconstitute 
    the image with the attribute y 
    """
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.encoder, self.decoder = enc_dec_model(params)
        self.params= params

    def encode(self, x):
        return self.encoder(x)

    def get_optimizers(self):
        return (self.opt,)

    def decode(self, z, y):
        """
        The decoder takes as input the concatenation of z and y along the column axis

        For some reason, the graph (eagerly mode = False) does not accept the numpy array in this method, so we will use the tensors
        """
        bs = y.shape[0]
        y = tf.cast(y, tf.float32)
        y = tf.expand_dims(y, axis = -1)
        y = tf.expand_dims(y, axis = -1)
        y = tf.repeat(y, 2, axis = -1)
        y = tf.repeat(y, 2, axis = 2)
        z = tf.reshape(z,(bs,512,2,2))
        zy = tf.concat((z,y), axis = 1)
        zy = tf.reshape(zy,(bs,2,2,512+len(self.params.get("ATTR"))))
        return self.decoder(zy)
        
    

    def call(self, x, y = None, mode = ''):
        z = self.encode(x)

        if y is None:
            return z
        
        if mode == 'decode':
            return self.decode(z,y)
        
        return z, self.decode(z, y)
    

    def predict(self, x, y):
        self.trainable = False
        z = self.encode(x)
        return self.decode(z,y)


class Fader(Model):
    def __init__(self, params):
        super(Fader, self).__init__()
        self.params = params
        self.ae = AutoEncoder(params)
        self.discriminator = discriminator(params)
        self.n_iter = 0
        self.lambda_dis = 0


    def get_optimizers(self):
        return (self.ae_opt, self.dis_opt)

    def compile(self, ae_opt, dis_opt, ae_loss, dis_loss=attr_loss, dis_metrics = tf.keras.metrics.BinaryAccuracy()):
        super(Fader,self).compile()
        self.ae_opt = ae_opt
        self.dis_opt = dis_opt
        self.dis_loss = dis_loss
        self.dis_metrics = dis_metrics
        self.ae_loss = ae_loss

    @tf.function
    def evaluate_on_val(self,data):
        x,y = data
        self.discriminator.trainable = False
        self.ae.trainable = False
        z, decoded = self.ae(x,y)
        y_preds = self.discriminator(z)

        #Discriminator
        dis_loss  = self.dis_loss(y, y_preds)
        dis_accuracy = self.dis_metrics(y, y_preds)

        # Autoencodeodr
        ae_loss = self.ae_loss(x, decoded)
        ae_loss += self.dis_loss(y, 1-y_preds)*self.lambda_dis

        return ae_loss, dis_loss, dis_accuracy

    @tf.function
    
    def  train_step(self,data):
        """
        This function can be applied using model.fit but we prefer to create our own custom training loop in main 
        (especially to have control over the loading of data and therefore the RAM memory)

        This method is the version of train_step customized to have total control over the training (especially the batch)
        """
        x,y = data
        #Training of the discriminator
        self.discriminator.trainable = True
        self.ae.trainable = False

        z = self.ae(x)
        with tf.GradientTape() as tape:
            y_preds = self.discriminator(z)
            dis_loss  = self.dis_loss(y, y_preds)
            dis_accuracy = self.dis_metrics(y, y_preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        #Training of the autoencdoer
        self.discriminator.trainable = False
        self.ae.trainable = True

        with tf.GradientTape() as tape:
            z, decoded = self.ae(x,y)
            dis_preds = self.discriminator(z)
            ae_loss = self.ae_loss(x, decoded)
            ae_loss += self.dis_loss(y, 1-dis_preds)*self.lambda_dis
        grads = tape.gradient(ae_loss, self.ae.trainable_weights)
        self.ae_opt.apply_gradients(zip(grads, self.ae.trainable_weights))

        self.n_iter+=1
        self.lambda_dis = 0.0001*min(self.n_iter/500000, 1)
        return ae_loss, dis_loss, dis_accuracy


# if __name__=="__main__":
#     enc, dec = enc_dec_model(params)
#     disc = discriminator(params)
#     classifier = Classifier(params)
#     enc.summary()
#     dec.summary()
#     disc.summary()
#     classifier.summary()