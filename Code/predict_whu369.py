
# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
from PIL import Image, ImageOps
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import random
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.models import load_model

import os
#os.system("sudo pip install PIL")


def get_input(images_paths):
    desired_size = 50
    imgs_array = []
    for image_path in images_paths:
        img = Image.open(image_path)
        old_size = img.size

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_img = Image.new('RGB', (desired_size, desired_size))
        new_img.paste(img, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

        new_img_array = np.asarray(new_img.convert('RGB'))

        imgs_array.append(new_img_array)

    return np.array(imgs_array)


def predict(x):

    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    x_test =  get_input(x)
    x_test = x_test.reshape(len(x_test), -1)
    x_test = x_test / 255

    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_whu369.hdf5')

    y_pred = np.argmax(model.predict(x_test), axis=1)

    return y_pred, model

