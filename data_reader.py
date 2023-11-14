import os
import numpy as np
import pathlib
import IPython.display as display
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow_datasets as tfd
import keras
import keras_cv
from keras import layers
from keras.models import Sequential
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split

# OUR IMPORTS:
from custom_functions import *

class DataReader:
    def __init__(self, X_path, y_path):
        # Paths
        self.X_train_masterpath = X_path
        self.y_train_masterpath = y_path
        self.X_paths = None
        self.y_paths = None
        # Data:
        self.train_ds = None
        self.val_ds = None
        self.image_size = (1024, 1024)

    def get_file_lists(self, train_X_masterpath = None, train_y_masterpath = None):
        # Ensure we have paths:
        if train_X_masterpath == None:
            train_X_masterpath = self.X_train_masterpath
        if train_y_masterpath == None:
            train_y_masterpath = self.y_train_masterpath
        # a list to collect paths of 1000 images
        train_X_paths = []
        X_paths_raw = os.walk(train_X_masterpath)
        for root, dirs, files in X_paths_raw:
            # iterate over 1000 images
            for file in files:
                # create path
                path = os.path.join(root,file)
                # Check to see if it's a pre-disaster picture
                if "pre" not in path:
                    continue
                # add path to list
                train_X_paths.append(path)
        # a list to collect paths of 1000 images
        train_y_paths = []
        y_paths_raw = os.walk(train_y_masterpath)
        for root, dirs, files in y_paths_raw:
            # iterate over 1000 images
            for file in files:
                # create path
                path = os.path.join(root,file)
                # Check to see if it's a pre-disaster picture AND to make sure it matches an input file:
                if "pre" not in path or f"{train_X_masterpath}" "\\" + path.split('\\')[-1][0:-11]+".png" not in train_X_paths:
                    continue
                # add path to list
                train_y_paths.append(path)

        # Sort the data so each point is in the same position:
        train_X_paths.sort()
        train_y_paths.sort()
        # Finalize:
        assert(len(train_X_paths) == len(train_y_paths))
        print(f"---\nX : {len(train_X_paths)} files | y: {len(train_y_paths)} files\n---")
        self.X_paths = train_X_paths
        self.y_paths = train_y_paths
        return train_X_paths, train_y_paths
    
    def get_tf_data(self, X_paths = None, y_paths = None, new_size = None, desired_amount = None):
        # Ensure we have paths:
        if X_paths == None:
            X_paths = self.X_paths
        if y_paths == None:
            y_paths = self.y_paths
        if desired_amount == None:
            desired_amount = len(X_paths)
        # Get tf object from each file:
        # Next, turn the files into points:
        X = []
        y = []

        for i in range(desired_amount): # WE RUN OUT OF RAM REALLY QUICKLY ON THIS...
            # Get the corresponding files:
            file_X = tf.io.read_file(X_paths[i])
            file_y = tf.io.read_file(y_paths[i])

            # Decode them into data:
            X.append(tf.image.decode_png(file_X, channels=3, dtype=tf.uint8))
            y.append(tf.image.decode_png(file_y, channels=1, dtype=tf.uint8))
        # Resizing:
        if new_size != None:
            X = [resize_image(i, new_size) for i in X]
            y = [resize_mask(m, new_size) for m in y]
            self.size = new_size
        
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

        print(f"---\n{len(train_X)} in train | {len(val_X)} in val\n---")
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_X, val_y))
        return self.train_ds, self.val_ds

        
if __name__ == "__main__":
    dr = DataReader("data/train/images", "data/train/targets")
    one, two = dr.get_file_lists()
    three, four = dr.get_tf_data()
    print(type(three), "\n", type(four))