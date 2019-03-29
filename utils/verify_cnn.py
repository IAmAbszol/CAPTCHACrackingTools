import numpy as np
import pandas as pd
import sys, os
import argparse
import random
import tensorflow as tf

from PIL import Image
from itertools import chain
from sklearn import model_selection
from sklearn import preprocessing
from matplotlib import pyplot as plt

import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, model_from_json

import helper

parser = argparse.ArgumentParser(description='Predicts character from segmented CAPTCHA image chosen')
parser.add_argument('-i', '--image', required=True, help='Selected segmented image.')
parser.add_argument('-m', '--model', required=True, help='Trained CNN model for use')
args = parser.parse_args()

# Load image from arguments passed
im = Image.open(args.image)

# Convert image to suitable format, 28x28 square
X = np.array(im.convert('L').resize((28, 34), Image.ANTIALIAS))
X = X.reshape(1, 28, 34, 1)
X = X.astype('float32')
X /= 255

# Prepare labels for inverse_transform
labels = helper.training_characters
label_encoder = preprocessing.LabelEncoder()
y_labeled = label_encoder.fit_transform(labels)

model = "{}.json".format(args.model)
weights = "{}.j5".format(args.model)

json_file = open(model, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights)

loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=.001), metrics=['accuracy'])
prediction = loaded_model.predict(X)
prediction = np.where(prediction > 0.5, 1, 0)

plt.imshow(im)
plt.text(0, 5, label_encoder.inverse_transform([np.argmax(prediction)]))
plt.show()
