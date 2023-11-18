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
from data_reader import DataReader

class StuffNet:

    def __init__(self, filename=None):
        if filename == None:
            self.model = self.make_base_model()
        else:
            self.model = keras.models.load_model(filename)
        
        return self.model
            
    
    def make_base_model(self):
        # Use pre-trained DenseNet121 without head
        base = keras.applications.DenseNet121(input_shape=[256,256,3],
                                            include_top=False,
                                            weights='imagenet')

        #final ReLU activation layer for each feature map size, i.e. 4, 8, 16, 32, and 64, required for skip-connections
        skip_names = ['conv1/relu', # size 64*64
                    'pool2_relu',  # size 32*32
                    'pool3_relu',  # size 16*16
                    'pool4_relu',  # size 8*8
                    'relu'        # size 4*4
                    ]

        #output of these layers
        skip_outputs = [base.get_layer(name).output for name in skip_names]
        # Build our downstack layers (where "downstack" means "max pooling")
        downstack = keras.Model(inputs=base.input,
                            outputs=skip_outputs)
        downstack.trainable = True # make sure we can train these

        # Four upstack layers for upsampling sizes
        # 4->8, 8->16, 16->32, 32->64
        upstack = [pix2pix.upsample(512,3),
                pix2pix.upsample(256,3),
                pix2pix.upsample(128,3),
                pix2pix.upsample(64,3)]

        # define the input layer
        inputs = keras.layers.Input(shape=[256,256,3])

        # downsample
        down = downstack(inputs)
        out = down[-1]

        # prepare skip-connections
        skips = reversed(down[:-1])
        # choose the last layer at first 4 --> 8

        # upsample with skip-connections
        for up, skip in zip(upstack,skips):
            out = up(out)
            out = keras.layers.Concatenate()([out,skip])

        # define the final transpose conv layer
        # image 128 by 128 with 59 classes
        out = keras.layers.Conv2DTranspose(1, 3,
                                        strides=2,
                                        padding='same',
                                        )(out)
        # complete unet model
        unet = keras.Model(inputs=inputs, outputs=out)

        self.model = unet
        return self.model

    # def weight_from_file(self, filename):
    #     old_model = keras.models.load_model("model3_training/model3.keras")

