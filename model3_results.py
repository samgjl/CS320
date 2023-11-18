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
import json

BATCH = 32
test_X_masterpath = "data/test/images"
test_y_masterpath = "data/test/targets"
dr_test = DataReader(test_X_masterpath, test_y_masterpath)
dr_test.get_file_lists_colab()
test, none = dr_test.get_tf_data(new_size = (256, 256), split = False)
test = test.batch(BATCH)

# get model
unet = keras.models.load_model("model3_training/model3.keras")
# prediction on test

results = unet.evaluate(test, batch_size = BATCH, verbose=1)
print(results)

# Get lists for plotting:
# m2training_file = open("model2_training/200EpochsHistory.txt", "r")
# model2_training = json.loads(m2training_file.read())
# train_loss = model2_training["loss"]
# train_acc = model2_training["accuracy"]
# val_loss = model2_training["val_loss"]
# val_acc = model2_training["val_accuracy"]
# Plot:
# X = np.arange(0, len(train_loss), 1)
# fig, axis = plt.subplots(1, 2)
# axis[0].plot(train_loss)
# axis[0].plot(val_loss)
# axis[1].plot(train_acc)
# axis[1].plot(val_acc)
# axis[0].set_xlabel("Epoch")
# axis[0].set_ylabel("Loss")
# axis[0].set_xlabel("Epoch")
# axis[0].set_ylabel("Accuracy")
# axis[0].legend(["Train", "Validation"])
# axis[1].legend(["Train", "Validation"])
# plt.savefig("model3_results/eval_over_time.png")
# plt.show()



NORM = mpl.colors.Normalize(vmin=0, vmax=2)
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
# plt.suptitle('Prediction After 20 Epochs (No Fine-tuning)', color='red', size=20)
plt.savefig("model3_results/result.png")
plt.show()