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

train_X_masterpath = "data/train/images"
train_y_masterpath = "data/train/targets"
# Get data:
dr = DataReader(train_X_masterpath, train_y_masterpath)
# dr.get_file_lists()
dr.get_file_lists_colab()
train, val = dr.get_tf_data(new_size = (256, 256), desired_amount=10)

#setting the batch size
BATCH = 32

AT = tf.data.AUTOTUNE
#buffersize
BUFFER = 1000

STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH

train = dr.augment(train)
train = train.cache().shuffle(BUFFER).batch(BATCH).repeat()
train = train.prefetch(buffer_size=AT)
val = val.batch(BATCH)

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

# compiling the model
def Compile_Model():
    unet.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=keras.optimizers.legacy.RMSprop(learning_rate=0.00025),
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
Compile_Model()

#training and fine-tuning
hist_1 = unet.fit(train,
               validation_data=val,
               steps_per_epoch=STEPS_PER_EPOCH,
               validation_steps=VALIDATION_STEPS,
               epochs=50,
               verbose=1)

# select a validation data batch
img, mask = next(iter(val))
# make prediction
pred = unet.predict(img)
plt.figure(figsize=(20,28))


hist = hist_1.history
precision = hist["precision"]
recall = hist["recall"]
val_prec = hist["val_precision"]
val_rec = hist["val_recall"]
hist["F1Score"] = [2*(precision[i]*recall[i]/(precision[i]+recall[i])) if (precision[i] + recall[i] != 0) else 0 for i in range(len(precision))]
hist["val_F1Score"] = [2*(val_prec[i]*val_rec[i]/(val_prec[i]+val_rec[i])) if (val_prec[i] + val_rec[i] != 0) else 0 for i in range(len(val_prec))]

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
plt.suptitle('Prediction After 200 Epochs', color='red', size=20)
plt.savefig("model3_training/result.png")

# Saving extras:
f = open("model3_training/200EpochsHistory.txt", "w")
f.write(str(hist).replace('\'', '\"'))
f.close
unet.save("model3_training/model3.keras")

plt.show()
