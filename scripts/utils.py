#!/usr/bin/env python
## Youcef Chorfi

import os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)+"/data")
import pickle
from Models import Classifier, Fader, AutoEncoder
# import glob


from data import getParams

params = getParams()


def denormalize(image):
    im = np.array(image)
    im = 127.5*(im + 1)
    return np.uint8(im)


def load_model_histroy(file, train = False, params=params):
    """
    Load a model
    -----  
    Parameters: 
    file: str, filename
    """
    path = str(Path(__file__).parent)+'/'+params.get("MODELS_PATH")
    model_params = np.load(path+'/'+file+'/params.npy', allow_pickle = True).item()
    history = np.load(path+'/'+file+'/history.npy', allow_pickle = True).item()

    model_type = file.split('_')[0]

    if model_type == 'classifier':
        model = Classifier(model_params, history)
    elif model_type =='fader':
        model = Fader(model_params)
    elif model_type =='ae':
        model = AutoEncoder(model_params)
    else: 
        raise ValueError(f"invalid model_type = {model_type}, possible value are 'classifier', 'fader' or 'ae'")

    model.load_weights(path+'/'+file+'/weights')
    model.trainable = train

    return model, history