import numpy as np
import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from data_reader import *
from model4 import *
import random
import json

# Get our model:
unet = keras.models.load_model("model4_training/model4_f1.keras")
# Data config:
BATCH = 32
test_X_masterpath = "data/test/images"
test_y_masterpath = "data/test/targets"
dr_test = DataReader(test_X_masterpath, test_y_masterpath)
dr_test.get_file_lists_colab()
test, test_X, test_y = dr_test.get_tf_data(new_size = (256, 256), test_data = True)
test = test.batch(BATCH)

# get model
unet = keras.models.load_model("model4_training/model4_f1.keras")
results = unet.evaluate(test, batch_size = BATCH, verbose=1)
print(results)


# Get lists for plotting:
m2training_file = open("model4_training/model4_f1_initial.txt", "r")
model_training = json.loads(m2training_file.read().replace('\'', '\"'))
train_loss = model_training["loss"]
train_acc = model_training["accuracy"]

val_loss = model_training["val_loss"]
val_acc = model_training["val_accuracy"]

prec = model_training["precision"]
val_prec = model_training["val_precision"]
recall = model_training["recall"]
val_rec = model_training["val_recall"]

f1 = np.array(prec) * np.array(recall) * 2 / (np.array(prec) + np.array(recall))
val_f1 = np.array(val_prec) * np.array(val_rec) * 2 / (np.array(val_prec) + np.array(val_rec))

# Plot:
X = np.arange(0, len(train_loss), 1)
fig, axis = plt.subplots(2, 3)
axis[0, 0].plot(train_loss)
axis[0, 0].plot(val_loss)

axis[0, 1].plot(train_acc)
axis[0, 1].plot(val_acc)

axis[1, 0].plot(prec)
axis[1, 0].plot(val_prec)

axis[1, 1].plot(recall)
axis[1, 1].plot(val_rec)

axis[1, 2].plot(f1)
axis[1, 2].plot(val_f1)

axis[0, 0].set_xlabel("Epoch")
axis[0, 1].set_xlabel("Epoch")
axis[1, 0].set_xlabel("Epoch")
axis[1, 1].set_xlabel("Epoch")
axis[1, 2].set_xlabel("Epoch")

axis[0, 0].set_ylabel("Loss")
axis[0, 1].set_ylabel("Accuracy")
axis[1, 0].set_ylabel("Precision")
axis[1, 1].set_ylabel("Recall")
axis[1, 2].set_ylabel("F1 Score")
axis[0, 0].legend(["Train", "Validation"])
axis[0, 1].legend(["Train", "Validation"])
axis[1, 0].legend(["Train", "Validation"])
axis[1, 1].legend(["Train", "Validation"])
axis[1, 2].legend(["Train", "Validation"])
plt.savefig("model4_results/eval_over_time.png")
plt.show()


