#!/usr/bin/env python
## Youcef Chorfi

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)

# Time for tensor board
from time import time

# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Reshape, ReLU, 
                                    LeakyReLU, Dropout, UpSampling2D, 
                                    BatchNormalization, Conv2DTranspose)




from data import getParams, Data_loader
from Models import *
import warnings
warnings.simplefilter("ignore")

params = getParams()
IMG_SIZE = eval(params.get('IMG_SIZE'))



# opt_ae = keras.optimizers.Adam(1e-4)
# opt_disc = keras.optimizers.Adam(1e-4)
# loss_fn = keras.losses.BinaryCrossentropy()

# for epoch in range(10):
#     for idx, real in enumerate(tqdm(dataset)):
#         batch_size = real.shape[0]
#         random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
#         fake = generator(random_latent_vectors)

#         if idx % 100 == 0:
#             img = keras.preprocessing.image.array_to_img(fake[0])
#             img.save(f"generated_images/generated_img{epoch}_{idx}_.png")

#         ### Train Discriminator: max log(D(x)) + log(1 - D(G(z))
#         with tf.GradientTape() as disc_tape:
#             loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
#             loss_disc_fake = loss_fn(tf.zeros(batch_size, 1), discriminator(fake))
#             loss_disc = (loss_disc_real + loss_disc_fake)/2

#         grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
#         opt_disc.apply_gradients(
#             zip(grads, discriminator.trainable_weights)
#         )

#         ### Train Generator min log(1 - D(G(z)) <-> max log(D(G(z))
#         with tf.GradientTape() as gen_tape:
#             fake = generator(random_latent_vectors)
#             output = discriminator(fake)
#             loss_gen = loss_fn(tf.ones(batch_size, 1), output)

#         grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
#         opt_gen.apply_gradients(
#             zip(grads, generator.trainable_weights)
#         )


if __name__=="__main__":
    enc, dec = enc_dec_model(params)
    disc = discriminator(params)
    classifier = Classifier(params)
    enc.summary()
    dec.summary()
    disc.summary()
    classifier.summary()