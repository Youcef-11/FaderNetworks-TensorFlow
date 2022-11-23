#!/usr/bin/env python
## Youcef Chorfi


import numpy as np
import tensorflow as tf
import os 
import pickle
from Models import Classifier, Fader, AutoEncoder
# import glob

# def vstack(array1, array2):
#     try:
#         new_array = np.vstack((array1,array2))
#     except ValueError:
#         if len(array1) == 0:
#             return array2
#         elif len(array2) == 0:
#             return array1

#     return new_array

# def hstack(array1, array2):
#     try:
#         new_array = np.hstack((array1,array2))
#     except ValueError:
#         if len(array1) == 0:
#             return array2
#         elif len(array2) == 0:
#             return array1

#     return new_array



# def normalize(image):    
#     # Normalization entre -1 et 1 
#     return image/127.5 -1

# def denormalize(image):
#     im = np.array(image)
#     im = 127.5*(im + 1)
#     return np.uint8(im)

# def save_model(model, name, folder_name= 'models', pickle_save = False):
#     if not  os.path.isdir(folder_name):
#         os.mkdir(folder_name)

#     print("Enrigistrement du model")
#     if pickle_save: 
#         filehandler = open(folder_name + '/' + name + '.pkl', 'wb')
#         pickle.dump(model, filehandler)
#         filehandler.close()
#     else:
#         model.save(folder_name + '/' + name)


def load_model(path, model_type, params_name = 'params.npy', weights_name ='weights' , train = False):
    """
    Charge un model, uniqnument pour l'inférence ce modèle ne peut pas etre entrainer étant donnée qu'on enrigistre pas le statut des optimizers 
    -----  
    Parameters : 
    path : str, chemin vers le dossier contenant le modèle
    param_name : str, nom du fichier contenant les paramètres du modele
    weights_name : str, nom du fichier contenantl les poids du modèle
    model_type : str, 'c' pour un classifier, 'f' pour un fader, 'ae' pour un autoencoder
    """
    params = np.load(path + '/'+ params_name, allow_pickle = True).item()
    if model_type == 'c':
        model = Classifier(params)
    elif model_type =='f':
        model = Fader(params)
    elif model_type =='ae':
        model = AutoEncoder(params)
    else: 
        raise ValueError(f"invalid model_type = {model_type}, possible value are 'c', 'f' or 'ae'")

    model.load_weights(path + '/' +weights_name)
    model.trainable = train

    return model


# def load_history(path : str, name = "history.npy", from_model_path= False):
#     if from_model_path:
#         index = path[::-1].index('/') 
#         path = path[:len(path) - index - 1]

#     if os.path.isfile(path +   '/'+ name):
#         h = np.load(path+ '/' + name , allow_pickle = True).item()
#         return h 

# def save_history(h:dict, path, name = "history.npy"):
#     if os.path.isfile(path +   '/'+ name) and len(h[h.keys()[0]]) != 1:
#         h1= np.load(path+ '/' + name , allow_pickle = True).item()
#         for p in h1:
#             h1[p].extend(h[p])
#         h = h1
#     np.save(path+ '/' + name, h)