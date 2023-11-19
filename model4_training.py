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

# Get data:
BUFFER = 1000
BATCH = 32
STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH
train, val = default_method()

# Make model:
image_dims = (256, 256)
model4 = Model4(image_dims=image_dims)
# model4.compile_bce()
# model4.compile_scce()
model4.verbatim()
unet = model4.model
# unet = keras.models.load_model("model4_training/model4_original.keras")
# unet.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0025),
#                     loss=keras.losses.CategoricalCrossentropy(),
#                     metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

# weights = {
#     0: 1, 
#     1: 5
# }
history = unet.fit(train,
               validation_data=val,
               steps_per_epoch=STEPS_PER_EPOCH,
               validation_steps=VALIDATION_STEPS,
               epochs=50,
            #    class_weight = weights,
               verbose=1)

# select a validation data batch
img, mask = next(iter(val))
# make prediction
pred = unet.predict(img)
plt.figure(figsize=(20,28))


NORM = mpl.colors.Normalize(vmin=-0.5, vmax=1.5)

k = 0
for i in pred:
    # plot the predicted mask
    plt.subplot(4,3,1+k*3)
    i = tf.argmax(i, axis=-1)
    plt.imshow(i,cmap='gray', norm=NORM)
    plt.axis('off')
    plt.title('Prediction')

    # plot the groundtruth mask
    plt.subplot(4,3,2+k*3)
    plt.imshow(mask[k], cmap='gray', norm=NORM)
    plt.axis('off')
    plt.title('Ground Truth')

    # plot the actual image
    plt.subplot(4,3,3+k*3)
    plt.imshow(img[k])
    plt.axis('off')
    plt.title('Actual Image')
    k += 1
    if k == 4: break
plt.suptitle('Training Results', color='red', size=20)
plt.savefig("model4_training/result.png")

# Saving extras:
f = open("model4_training/200EpochsHistory.txt", "w")
f.write(str(history.history))
f.close
unet.save("model4_training/model4.keras")

plt.show()




