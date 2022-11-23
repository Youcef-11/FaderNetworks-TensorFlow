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
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

# Time for tensor board
from time import time

from tensorflow.keras import Model
# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Reshape, ReLU, 
                                    LeakyReLU, Dropout, UpSampling2D, 
                                    BatchNormalization, Conv2DTranspose, Layer)




from data import getParams, Data_loader
import warnings
warnings.simplefilter("ignore")

params = getParams()
IMG_SIZE = eval(params.get('IMG_SIZE'))

@tf.function
def compute_accuracy(yt,yp):
    count = 0
    for i in range(len(yp)):
        if tf.argmax(yp[i]) == tf.argmax(yt[i]):
            count+=1
    return count/len(yp) 

@tf.function
def attr_loss_accuracy(y_true, y_preds, loss_funtion = tf.nn.softmax_cross_entropy_with_logits):
    '''
    Computes softmax cross entropy between logits and labels.

    Measures the probability error in discrete classification tasks in which 
    the classes are mutually exclusive (each entry is in exactly one class).

    Here we have a Multi-Label classification tast which each label is coded in 2 bits
    [1,0] : False or [0,1] : True 

    Args:
    y_true (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS, 2)
    y_preds (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS, 2)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
        accuracy (scalar Tensor): value of the accuracy function for the batch
    '''
    bs = y_true.shape[0]
    n_attr = y_true.shape[1]
    loss = 0
    accuracy = []

    for i in range(0,n_attr):
        yt = y_true[:,i]
        yp = y_preds[:,i]
        loss += tf.reduce_sum(loss_funtion(yt,yp))/bs
        accuracy.append(compute_accuracy(yt, yp))


    return loss, tf.reduce_mean(accuracy)


def save_model_weights(model, name, folder_name ='models', get_optimizers = False):
    '''
    Save model weights and params
    '''
    print("Saving Network Weights...")
    model.save_weights(folder_name + '/' + name + '/' + 'weights')
    np.save(folder_name + '/' + name + '/' + 'params', model.params)


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
                        ReLU()
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
                                Dense(len(params.get("ATTR")))
                               ],
                               name = "discriminator")
    return discriminator


class Classifier(Model):
    '''
    Multi-label Classification
    Input : Images of shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)
    Output : y_preds (float32 Tensor) probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS, 2)
    '''
    def __init__(self, params):
        super(Classifier, self).__init__()
        self.params = params
        self.model, _ = enc_dec_model(params)
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Dense(2*len(params.get("ATTR"))))
        self.model.add(Reshape((len(params.get("ATTR")),2)))
        # self.build((None, 256,256,3))

    def get_optimizers(self):
        return (self.opt,)



    def compile(self, optimizer, loss_acc =attr_loss_accuracy):
        super(Classifier, self).compile()
        self.opt = optimizer
        self.loss_acc = loss_acc

    @tf.function
    def eval_on_batch(self, data):
        x,y = data

        self.model.trainable = False
        y_preds = self.model(x)
        loss, acc= self.loss_acc(y, y_preds)
        return loss, acc


    @tf.function
    def train_step(self, data):
        x,y = data
        self.model.trainable = True
        with tf.GradientTape() as tape:
            y_preds = self(x)
            loss, acc= self.loss_acc(y, y_preds)
        grads = tape.gradient(loss, self.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.trainable_weights))

        return loss, acc
    
    # @tf.function
    # def eval_on_recons_attributes_batch(self, data, fader):
    #     """
    #     Le but de cette fonction est d'évaluer les performances du sur des images reconstruits avec des attibuts "hasardeux"
    #     """
    #     x,y = data
    #     y_const = tf.identity(y)
        
    #     fader.trainable = False
    #     self.trainable = False
    #     # On verifie que les attributs pour le fader sont des attributs pour lesquels le classifier est entrainé
    #     assert np.isin(fader.params.get("ATTR"), params.get("ATTR")).all()

    #     z = fader.ae(x)
    #     #Faire correspondre les attributs du classifier avec ceux du fader
    #     loss = []
    #     acc = []
    #     bs = y.shape[0]
    #     for i,attr in enumerate(fader.params.get("ATTR")):
    #         j = 2*i
    #         clf_ind = 2*self.params.get("ATTR").index(attr)
    #         for v in range(2):
    #             y.assign(y_const) 
    #             y[:, j:j+2].assign(tf.zeros((bs,2))) # = 0
    #             y[:, j+v].assign(tf.ones((bs))) # = 1
    #             # y_copy = y.numpy()
    #             # y_copy[:, j:j+2] = 0
    #             # y_copy[:, j+v] = 1
    #             output = fader.ae.decode(z, y_const)
    #             if v == 0 : 
    #                 clf_preds = self(output)[:, clf_ind : clf_ind +2]
    #             else:
    #                 clf_preds = tf.concat((clf_preds, self(output)[:, clf_ind: clf_ind +2]), 1)
    #         l, a = attr_loss_accuracy(y, clf_preds)
    #         loss.append(l)
    #         acc.append(a)


    #     fader.trainable = True
    #     return tf.reduce_mean(loss), tf.reduce_mean(acc)
    
    def fit(self, Data_train, Data_validation) :
        history = {'train_loss' : [], 'train_acc': [], 'val_loss' : [], 'val_acc' : []}
        best_acc = 0

        n_epoch = params.get("N_EPOCH")
        for epoch in range(n_epoch):
            print("Training...")
            loss = []
            acc = []
            for step, batch in enumerate(Data_train) :
                t = time()
                batch_x, batch_y  = batch
                l,a = self.train_step((batch_x, batch_y))

                loss.append(l)
                acc.append(a)
                print(f"epoch : {1 + epoch}/{n_epoch}, batch : {1 + step}/{Data_train.batch_number} : ", colored(f"acc = {a.numpy():.3f}, ","green"), colored(f"loss = {l.numpy():.3f}", "red"), "calculé en :", colored(f"{time() - t:.3f}s", "yellow"),end="\r")
                    
            history['train_loss'].append(np.mean(loss))
            history['train_acc'].append(np.mean(acc))

            #Eval loop 
            loss = []
            acc = []
            print("Evaluating...:")
            rand = random.choice(range(Data_validation.batch_number))
            for step in range(params.get("BATCH_SIZE")):
                t = time()

                batch_x, batch_y = Data_validation[rand]
                l, a= self.eval_on_batch((batch_x, batch_y))
                loss.append(l)
                acc.append(a)

            print(colored(f"val_acc = {np.mean(acc):.3f}, ","green"), colored(f"val_loss = {np.mean(loss):.3f}", "red"), "calculé en :", colored(f"{time() - t:.3f}s", "yellow"))
            
            # Peut nous permettre de tracer un graph.
            history['val_loss'].append(np.mean(loss))
            history['val_acc'].append(np.mean(acc))

            if history['val_acc'][-1] > best_acc: 
                best_acc = history['val_acc'][-1]
                save_model_weights(self, name = 'classifier', folder_name=params.get("CLASSIFIER_PATH"))
            
        return history

    def evaluate(self, Data_test) :
        os.system('clear')
        #Eval loop 
        loss = []
        acc = []
        print("*"*20+"Evaluation:"+"*"*20)
        for step, batch in enumerate(Data_test):
            t = time()

            batch_x, batch_y = batch
            l, a= self.eval_on_batch((batch_x, batch_y))
            loss.append(l)
            acc.append(a)

            print(f"batch : {1 + step}/{Data_test.batch_number} : ", colored(f"val_accuracy = {a.numpy():.2f}", "green"), colored(f"val_loss = {l.numpy():.2f}", "red"), "calculé en :", colored(f"{time() - t:.2f}s", "yellow"),end="\r")
        
        
        print(colored(f"mean_val_accuracy = {np.mean(acc):.2f}, ","green"), colored(f"val_loss = {np.mean(loss):.2f}", "red"))

    def call(self, x):
        x = self.model(x)
        return x

class AutoEncoder(Model):
    """
    La présence de cette classe est du au fait que le decoder a besoin de la represéntation latente z, et des attributs y pour reconstituer l'image avec l'attribut y 
    """
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.encoder, self.decoder = enc_dec_model(len(params.get("ATTR")))
        self.params= params

    def encode(self, x):
        return self.encoder(x)

    def get_optimizers(self):
        return (self.opt,)

    def decode(self, z, y):
        # Le décodeur prend en entrée la concaténation de z et de y selon l'axe des colones

        # Pour certaine raison, le graph (eagerly mode = False) n'accepte pas le numpy array dans cette methode, on utilisera alors les tenseurs
        y = tf.expand_dims(y, axis = 1)
        y = tf.expand_dims(y, axis = 2)
        y = tf.repeat(y, 2, axis = 1)
        y = tf.repeat(y, 2, axis = 2)
        # y = np.expand_dims(y,(1, 2))
        # y = np.repeat(y, 2, axis = 1)
        # y = np.repeat(y, 2, axis = 2)
        zy = tf.concat((z,y), axis = -1)
        return self.decoder(zy)
        
    

    def call(self, x, y = None, mode = ''):
        z = self.encode(x)

        if y is None:
            return z
        
        if mode == 'decode':
            return self.decode(z,y)
        
        return z, self.decode(z, y)


class Fader(Model):
    def __init__(self, params):
        super(Fader, self).__init__()
        self.params = params
        self.ae = AutoEncoder(params)
        self.discriminator = discriminator(params.n_attr)
        self.n_iter = 0
        self.lambda_dis = 0
    

    # def set_optimizer_weights(self,weights):
    #     self.ae_opt.set_weights(weights[0])
    #     self.dis_opt.set_weights(weights[1])


    def get_optimizers(self):
        return (self.ae_opt, self.dis_opt)

    def compile(self, ae_opt, dis_opt, ae_loss, dis_loss = attr_loss_accuracy):
        super(Fader,self).compile()
        self.ae_opt = ae_opt
        self.dis_opt = dis_opt
        self.dis_loss = dis_loss
        self.ae_loss = ae_loss

    @tf.function
    def evaluate_on_val(self,data):
        x,y = data
        self.discriminator.trainable = False
        self.ae.trainable = False
        z, decoded = self.ae(x,y)
        y_preds = self.discriminator(z)

        #Discriminator
        dis_loss, dis_accuracy = self.dis_loss(y, y_preds)

        # Autoencodeodr
        ae_loss = self.ae_loss(x, decoded)
        ae_loss = ae_loss + dis_loss*self.lambda_dis

        return ae_loss, dis_loss, dis_accuracy

    @tf.function
    # Cette fonction peut s'apperler en utilisant model.fit mais on a préférer créer notre boucle d'entrainement personalisé danas main (notamment pour avoir le controle sur le chargement des données et donc la mémoire RAM)
    def  train_step(self,data):
        """
        Cette méthode est la version de train_step customisée pour avoir le controole total sur le training (notamment les batch)
        """
        x,y = data
        #Training of the discriminator
        self.discriminator.trainable = True
        self.ae.trainable = False

        z = self.ae(x)
        with tf.GradientTape() as tape:
            y_preds = self.discriminator(z)
            dis_loss ,dis_accuracy = self.dis_loss(y, y_preds)

        grads = tape.gradient(dis_loss, self.discriminator.trainable_weights)
        self.dis_opt.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        #Training of the autoencdoer
        self.discriminator.trainable = False
        self.ae.trainable = True

        with tf.GradientTape() as tape:
            z, decoded = self.ae(x,y)
            dis_preds = self.discriminator(z)
            ae_loss = self.ae_loss(x, decoded)
            ae_loss = ae_loss + self.dis_loss(y, dis_preds)[0]*self.lambda_dis
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