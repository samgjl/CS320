import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from data_reader import *
from model4 import *

# Get data:
BUFFER = 1000
BATCH = 32
STEPS_PER_EPOCH = 800//BATCH
VALIDATION_STEPS = 200//BATCH
train, val = default_method()

# Make model:
image_dims = (256, 256)
# model4 = Model4(image_dims=image_dims)
# model4.compile_bce()
# model4.compile_scce()
# model4.compile_f1()
# unet = model4.model
unet = keras.models.load_model("model4_training/model4_f1.keras")
unet.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0001),
            loss=f1_loss,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

history = unet.fit(train,
               validation_data=val,
               steps_per_epoch=STEPS_PER_EPOCH,
               validation_steps=VALIDATION_STEPS,
               epochs=20,
               verbose=1)

# Saving extras:
f = open("model4_training/model4_f1_history.txt", "w")
f.write(str(history.history))
f.close
unet.save("model4_training/model4_f1_second_pass.keras")