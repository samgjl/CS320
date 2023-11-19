import os
import numpy as np
import pathlib
import IPython.display as display
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
# import tensorflow_datasets as tfd
import keras
import keras_cv
from keras import layers
from keras.models import Sequential
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
import importlib.util
import sys
from data_reader import *
from model4 import Model4
import random

unet = keras.models.load_model("model4_training/model4.keras")


train, val = default_method(desired_amount=10)
# select a validation data batch
img, mask = next(iter(val))
# make prediction
pred = unet.predict(img)
plt.figure(figsize=(20,28))


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Predicted mask', 'True mask', 'Input image']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig("model4_results/result.png")
    # plt.show()
  
i = random.randint(0, len(img))
sample_image = img[i]
sample_mask = mask[i]
prediction = unet.predict(sample_image[tf.newaxis, ...])[0]
predicted_mask = (prediction > 0.5).astype(np.uint8)
display([predicted_mask, sample_mask, sample_image])