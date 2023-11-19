# This is intended to display our loss and accuracy as functions over the number of epochs.
# This also tests our data on  XView2's provided "test" dataset.
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from data_reader import DataReader
import json

# Get the data:
BATCH = 32
test_X_masterpath = "data/test/images"
test_y_masterpath = "data/test/targets"
dr_test = DataReader(test_X_masterpath, test_y_masterpath)
dr_test.get_file_lists_colab()
test, test_X, test_y = dr_test.get_tf_data(new_size = (256, 256), test_data = True)
test = test.batch(BATCH)

# get model
unet = keras.models.load_model("model2_training/model2.keras")
# prediction on test
unet.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.legacy.RMSprop(learning_rate=0.00025),
            metrics=['accuracy'])
results = unet.evaluate(test, batch_size = BATCH, verbose=1)
print(results)


# Get lists for plotting:
m2training_file = open("model2_training/200EpochsHistory.txt", "r")
model2_training = json.loads(m2training_file.read())
train_loss = model2_training["loss"]
train_acc = model2_training["accuracy"]
val_loss = model2_training["val_loss"]
val_acc = model2_training["val_accuracy"]
# Plot:
X = np.arange(0, len(train_loss), 1)
fig, axis = plt.subplots(1, 2)
axis[0].plot(train_loss)
axis[0].plot(val_loss)
axis[1].plot(train_acc)
axis[1].plot(val_acc)
axis[0].set_xlabel("Epoch")
axis[0].set_ylabel("Loss")
axis[0].set_xlabel("Epoch")
axis[1].set_ylabel("Accuracy")
axis[0].legend(["Train", "Validation"])
axis[1].legend(["Train", "Validation"])
plt.savefig("model2_results/eval_over_time.png")
plt.show()

img, mask = next(iter(test))
# make prediction
pred = unet.predict(img)
plt.figure(figsize=(20,28))
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
plt.savefig("model2_results/result.png")
plt.show()