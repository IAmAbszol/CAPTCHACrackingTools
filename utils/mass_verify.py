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

# Globals
captchas_cracked = 0
captchas_total = 0
characters_cracked = 0
characters_total = 0

parser = argparse.ArgumentParser(description='Predicts character from segmented CAPTCHA image chosen')
parser.add_argument('-f', '--file', required=True, help='CSV of images to verify.')
parser.add_argument('-m', '--model', required=True, help='Trained CNN model for use')
args = parser.parse_args()

# Load the data from the CSV
data = pd.read_csv(args.file, converters={"x": lambda z: z.strip("[]").split(", "), "w": lambda y: y.strip("[]").split(", "), "h": lambda x: x.strip("[]").split(", ")}, sep=',')
data = np.array(data.dropna())

# Prepare labels for inverse_transform
labels = helper.training_characters
y_labeled = label_encoder.fit_transform(labels)

model = "{}.json".format(args.model)
weights = "{}.j5".format(args.model)

json_file = open(model, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights)

loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=.001), metrics=['accuracy'])

for row in data:
	image = Image.open(row[1])
	if not image is None:

		if captchas_total % 100 == 0 and captchas_total != 0:
			print("Completed {}/{}.".format(captchas_total, len(data)))
	
		text = row[2]
		captcha_guess = []
	
		# Extract characters from image
		for index, p in enumerate(row[5]):
			char_im = image.crop((int(p), int(row[6]), int(row[4][index]), int(row[3][index])))
			char = row[2][index]
			
			# Convert image to suitable format, 28x28 square
			X = np.array(char_im.convert('L').resize((28, 28), Image.ANTIALIAS))
			X = X.reshape(1, 28, 28, 1)
			X = X.astype('float32')
			X /= 255

			prediction = loaded_model.predict(X)
			prediction = np.where(prediction > 0.5, 1, 0)
			
			cap = str(label_encoder.inverse_transform([np.argmax(prediction)])[0])
			captcha_guess.append(''.join(cap))			

			if cap == char:
				characters_cracked += 1
			characters_total += 1

		if text == ''.join(captcha_guess):
			captchas_cracked += 1
		captchas_total += 1
			
print("Crack Rate: {}/{} --> {}".format(captchas_cracked, captchas_total, (captchas_cracked/captchas_total)))
print("Character Crack Rate: {}/{} --> {}".format(characters_cracked, characters_total, (characters_cracked/characters_total)))
