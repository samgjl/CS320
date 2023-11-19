# This code lightly follows the sctucture of U-Net as described in Derrick Mwiti's tutorial on Creating U-Net from scratch.
# * This code specifically references the layer structure of the contracting and expanding paths.
# Linked here: https://www.machinelearningnuggets.com/image-segmentation-with-u-net-define-u-net-model-from-scratch-in-keras-and-tensorflow/
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

# Using F1 as a loss function inspired by Michal Haltuf's blog post in Kaggle.
# Linked here: https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric
@keras.saving.register_keras_serializable("CS320_final_project")
def f1_loss(y_true, y_pred):
    # True/False & Positive/Negative
    true_pos = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    true_neg = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    false_pos = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    false_neg = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) 
    # Get precision and Recall scores:
    precision = true_pos / (true_pos + false_pos + K.epsilon())
    recall = true_pos / (true_pos + false_neg + K.epsilon())
    # Get F1 Score:
    f1 = 2*precision*recall / (precision+recall+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1) # replace all "0 / 0" values with 0
    f1 = K.mean(f1) # Mean all f1 scores together to get a unified loss
    loss = 1 - f1 # A "good" F1 score is >0.7, so we're looking for <0.3 loss where possible (if we don't modify our tp/fn penalties)
    return loss 


class Model4:
    def __init__(self, image_dims = (256,256), safe = False):
        if not safe:
            self.make_base_model()
        else:
            self.make_base_model_one_function()
        
        self.model = keras.Model(inputs=self.input, outputs=self.output)

    def compile_bce(self, lr = 0.001):
        self.model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr),
                    loss=keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    def compile_f1(self, lr = 0.001):
        self.model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr),
                    loss=f1_loss,
                    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    def compile_scce(self, lr = 0.001):
        self.model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    # This function will make a U-Net style model
    def make_base_model(self, image_dims = (256, 256)):
        # Input layer:
        self.input = keras.layers.Input(shape=[image_dims[0], image_dims[1], 3])
        # Contracting Path:
        contract1, max_pool1 = self.contracting_layer(filters=16, previous_layer=self.input)
        contract2, max_pool2 = self.contracting_layer(filters=32, previous_layer=max_pool1)
        contract3, max_pool3 = self.contracting_layer(filters=64, previous_layer=max_pool2)
        contract4, max_pool4 = self.contracting_layer(filters=128, previous_layer=max_pool3)
        # Bottom of the U:
        lowest_layer = self.lowest_layer(filters=256, previous_layer=max_pool4)
        # Expanding Path:
        upscale4 = self.expanding_layer(filters=128, previous_layer=lowest_layer, match_layer=contract4)
        upscale3 = self.expanding_layer(filters=64, previous_layer=upscale4, match_layer=contract3)
        upscale2 = self.expanding_layer(filters=32, previous_layer=upscale3, match_layer=contract2)
        # upscale1 = self.expanding_layer(filters=16, previous_layer=upscale2, match_layer=activation1)
        # Expanding path (1):
        upscale1 = keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(upscale2)
        upscale1 = keras.layers.concatenate([upscale1, contract1], axis=3) # Why is this different ??????????????????
        upscale1 = keras.layers.BatchNormalization()(upscale1)
        upscale1 = keras.layers.Activation("ReLU")(upscale1)
        # Output layer:
        self.output = keras.layers.Conv2D(1, (1,1), activation="sigmoid")(upscale1) # We want this to be sigmoid so we get binary output

        return self.input, self.output

    def contracting_layer(self, filters = 1, previous_layer = None):
        contract1 = keras.layers.Conv2D(filters, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(previous_layer) # Uses he_normal initializer (arxiv.org/abs/1502.01852)
        contract1 = keras.layers.Dropout(0.1)(contract1) # Dropout layer
        contract1 = keras.layers.Conv2D(filters, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(contract1)
        batch_norm1 = keras.layers.BatchNormalization()(contract1)
        activation1 = keras.layers.Activation("ReLU")(batch_norm1) # ReLU activation layer (layers.Relu() also works)
        max_pool1 = keras.layers.MaxPooling2D((2,2))(activation1) # Max pooling layer
        return contract1, max_pool1 

    def lowest_layer(self, filters = 1, previous_layer = None):
        # Lowest Layer:
        lowest_layer = keras.layers.Conv2D(filters, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(previous_layer)
        batch_lowest = keras.layers.BatchNormalization()(lowest_layer) # Batch normalization layer
        activation_lowest = keras.layers.Activation("ReLU")(batch_lowest) # ReLU activation layer
        lowest_layer = keras.layers.Dropout(0.1)(activation_lowest) # Dropout layer
        # This is the end of the bottom of the U:
        lowest_layer = keras.layers.Conv2D(filters, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(lowest_layer)
        return lowest_layer

    
    def expanding_layer(self, filters = 1, previous_layer = None, match_layer = None):
        upscale = keras.layers.Conv2DTranspose(filters, (2,2), strides=(2,2), padding="same")(previous_layer) # Upsampling layer
        upscale = keras.layers.concatenate([upscale, match_layer]) # Concatenate to the corresponding place in the U
        upscale = keras.layers.BatchNormalization()(upscale) # Batch normalization layer
        upscale = keras.layers.Activation("ReLU")(upscale) # ReLU activation layer
        return upscale
    
    def make_base_model_one_function(self, image_dims = (256, 256)):
        # Input layer:
        self.input = keras.layers.Input(shape=[image_dims[0], image_dims[1], 3])
        # Contracting path (1):
        contract1 = keras.layers.Conv2D(16, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(self.input) # Uses he_normal initializer (arxiv.org/abs/1502.01852)
        contract1 = keras.layers.Dropout(0.1)(contract1) # Dropout layer
        contract1 = keras.layers.Conv2D(16, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(contract1)
        batch_norm1 = keras.layers.BatchNormalization()(contract1)
        activation1 = keras.layers.Activation("ReLU")(batch_norm1) # ReLU activation layer (layers.Relu() also works)
        max_pool1 = keras.layers.MaxPooling2D((2,2))(activation1) # Max pooling layer
        # Contracting path (2):
        contract2 = keras.layers.Conv2D(32, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(max_pool1)
        contract2 = keras.layers.Dropout(0.1)(contract2) 
        contract2 = keras.layers.Conv2D(32, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(contract2)
        batch_norm2 = keras.layers.BatchNormalization()(contract2)
        activation2 = keras.layers.Activation("ReLU")(batch_norm2)
        max_pool2 = keras.layers.MaxPooling2D((2,2))(activation2)
        # Contracting path (3):
        contract3 = keras.layers.Conv2D(64, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(max_pool2)
        contract3 = keras.layers.Dropout(0.1)(contract3)
        contract3 = keras.layers.Conv2D(64, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(contract3)
        batch_norm3 = keras.layers.BatchNormalization()(contract3)
        activation3 = keras.layers.Activation("ReLU")(batch_norm3)
        max_pool3 = keras.layers.MaxPooling2D((2,2))(activation3)
        # Contracting path (4):
        contract4 = keras.layers.Conv2D(128, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(max_pool3)
        contract4 = keras.layers.Dropout(0.1)(contract4)
        contract4 = keras.layers.Conv2D(128, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(contract4)
        batch_norm4 = keras.layers.BatchNormalization()(contract4)
        activation4 = keras.layers.Activation("ReLU")(batch_norm4)
        max_pool4 = keras.layers.MaxPooling2D((2,2))(activation4)
        # Contracting path (5) | BOTTOMING OUT HERE:
        lowest_layer = keras.layers.Conv2D(256, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(max_pool4)
        batch_lowest = keras.layers.BatchNormalization()(lowest_layer) # Batch normalization layer
        activation_lowest = keras.layers.Activation("ReLU")(batch_lowest) # ReLU activation layer
        lowest_layer = keras.layers.Dropout(0.1)(activation_lowest) # Dropout layer
        # This is the end of the bottom of the U:
        lowest_layer = keras.layers.Conv2D(256, (3,3), activation="ReLU", kernel_initializer=keras.initializers.HeNormal(), padding="same")(lowest_layer)
        # Expanding path (4):
        upscale4 = keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(lowest_layer) # Upsampling layer
        upscale4 = keras.layers.concatenate([upscale4, contract4]) # Concatenate to the corresponding place in the U
        upscale4 = keras.layers.BatchNormalization()(upscale4) # Batch normalization layer
        upscale4 = keras.layers.Activation("ReLU")(upscale4) # ReLU activation layer
        # Expanding path (3):
        upscale3 = keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(upscale4)
        upscale3 = keras.layers.concatenate([upscale3, contract3])
        upscale3 = keras.layers.BatchNormalization()(upscale3)
        upscale3 = keras.layers.Activation("ReLU")(upscale3)
        # Expanding path (2):
        upscale2 = keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(upscale3)
        upscale2 = keras.layers.concatenate([upscale2, contract2])
        upscale2 = keras.layers.BatchNormalization()(upscale2)
        upscale2 = keras.layers.Activation("ReLU")(upscale2)
        # Expanding path (1):
        upscale1 = keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(upscale2)
        upscale1 = keras.layers.concatenate([upscale1, contract1], axis=3)
        upscale1 = keras.layers.BatchNormalization()(upscale1)
        upscale1 = keras.layers.Activation("ReLU")(upscale1)
        
        # Output layer:
        self.output = keras.layers.Conv2D(2, (1,1), activation="sigmoid")(upscale1) # We want this to be sigmoid so we get binary output

        
            


if __name__ == "__main__":
    test_model = Model4(safe = False)
    test_model.model.summary()

