'''
	train_fast.py
	A live attempt at training small batches,
	exporting the model, and retrain based
	on new data generated.
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import random
import numpy as np
import argparse
import datetime
import tensorflow as tf
import keras

from PIL import Image, ImageFile
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Convolution2D, MaxPool2D
from keras.models import Sequential
from keras.models import model_from_json

from modified_claptcha import Claptcha
from modified_image import ImageCaptcha
from utils import helper

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Generate:

	def __init__(self, args):
		self.Characters = helper.training_characters
		self.Fonts = ["fonts/{}".format(i) for i in os.listdir(args['fonts']) if ".ttf" in i.lower() or ".otf" in i.lower()]

	"""
		The following are constants when it comes to input --> output, processing is different.
		If you build or place in your own module that you want to be trained upon,
		append or change generate_batch -> captcha_choices.
		
		The return of the function must be of the image and a list of offsets being delta x within
		the image. View modify_*.py to understand how I hacked the module to fit my needs.
	"""
	def generate_claptcha(self, word, length):
		# Claptcha captcha storage
		c = Claptcha(word, "{}".format(self.Fonts[random.randint(0, (len(self.Fonts))-1)]), resample=Image.BICUBIC, noise=(random.randint(0, 4) / 10))
		text, image, offsets = c.image

		return image, list(zip(offsets[0], offsets[2]))

	def generate_captcha(self, word, length):
		img_captcha = ImageCaptcha(width=40*int(length), height=70, fonts=self.Fonts)
		data, offsets = img_captcha.generate(word)
		data.seek(0)
		image = Image.open(data)
		captcha_offsets = [(offset[0], (offset[1] - offset[0])) for offset in offsets]

		return image, captcha_offsets

	def generate_batch(self, batch_size=1024, length=6):
		"""
			Generates a set batch size of batch size of the captcha's.
			:param batch_size: Set size to be generated.
			:param length: Length of CAPTCHA to be generated. Default is sufficient.
			:return: Returns Image list batch tupled as Image, Label.
		"""
		start = datetime.datetime.now()
		image_labels = []
		image_list = []
		text_list = []
		captcha_choices = [self.generate_claptcha]
		#captcha_choices = [self.generate_captcha, self.generate_claptcha]
		batch = 0
		choice = 0
		while batch < int(batch_size):
			if choice >= len(captcha_choices):
				choice = 0
			if batch != 0 and batch % int(batch_size / 10) == 0:
				print("generate_batch: Completed {}/{} - {}.".format(batch, batch_size, (datetime.datetime.now() - start)))

			word = ''.join([str(self.Characters[random.randrange(0, len(self.Characters))]) for i in range(int(length))])

			image, offsets = captcha_choices[choice](word, length)

			for idx, offset in enumerate(offsets):
				sub_image = image.crop((offset[0], 0, offset[0] + offset[1], image.size[1]))
				image_labels.append((sub_image, word[idx]))
			image_list.append(image)
			text_list.append(word)
			batch += 1
			choice += 1

		return image_labels

class Train:

	def __init__(self, args):
		self.batch_size = args['batchsize']
		self.generations = args['generations']
		self.length = 6
		self.generate = Generate(args)

	def generate_and_train(self):

		loaded_model = None

		# Load our model if it exists.
		if os.path.isdir('models/'):
			# Load Character Recognition Model
			print(os.getcwd())
			model = "{}.json".format("models/cnn_model")
			weights = "{}.j5".format("models/cnn_model")

			json_file = open(model, "r")
			loaded_model_json = json_file.read()
			json_file.close()
			loaded_model = model_from_json(loaded_model_json)
			loaded_model.load_weights(weights)

			loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=.001), metrics=['accuracy'])
			
		for sub_generation in range(int(self.generations)):

			image_labels = self.generate.generate_batch(batch_size=int(self.batch_size))
			X, y = list(zip(*image_labels))
			X = list(X)
			y = list(y)

			label_encoder = preprocessing.LabelEncoder()
			label_encoder.fit(helper.training_characters)
			y = [label_encoder.transform([v])[0] for v in y]
			y_encoded = keras.utils.to_categorical(y)

			width = 28
			height = 34

			for index, image in enumerate(X):
				X[index] = image.convert('L')
				X[index] = np.array(X[index].resize((width, height), Image.ANTIALIAS))

			X = np.array(X)

			X = X.reshape(len(X), width, height, 1)
			X = X.astype('float32')
			X /= 255

			# Splitting the train and test data
			X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_encoded, test_size=.2, random_state=5)

			batch_size = 128
			classes = len(y_encoded[0])
			epochs = 30
			
			if loaded_model is None:

				input = (X.shape[1], X.shape[2], 1)
				'''
				loaded_model = Sequential()
				loaded_model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input))
				loaded_model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
				loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
				loaded_model.add(Dropout(0.25))
				loaded_model.add(Flatten())
				loaded_model.add(Dense(128, activation='relu'))
				loaded_model.add(Dropout(0.5))
				loaded_model.add(Dense(classes, activation='softmax'))
				loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
				'''
				loaded_model = Sequential()
				kernel_size = (3,3)
				ip_activation = 'relu'
				ip_conv_0 = Conv2D(filters=32, kernel_size=kernel_size, input_shape=input, activation=ip_activation)
				loaded_model.add(ip_conv_0)
				
				ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernel_size, activation=ip_activation)
				loaded_model.add(ip_conv_0_1)
				
				pool_0 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')
				loaded_model.add(pool_0)
				
				ip_conv_1 = Conv2D(filters=64, kernel_size=kernel_size, activation=ip_activation)
				loaded_model.add(ip_conv_1)
				ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernel_size, activation=ip_activation)
				loaded_model.add(ip_conv_1_1)
				
				pool_1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')
				loaded_model.add(pool_1)
				
				drop_layer_0 = Dropout(.2)
				loaded_model.add(drop_layer_0)
				
				flat_layer_0 = Flatten()
				loaded_model.add(Flatten())
				
				h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
				loaded_model.add(h_dense_0)
				
				h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
				loaded_model.add(h_dense_1)
				
				op_activation = 'softmax'
				output_layer = Dense(units=classes, activation=op_activation, kernel_initializer='uniform')
				loaded_model.add(output_layer)
				
				opt = 'adam'
				loss = 'categorical_crossentropy'
				metrics = ['accuracy']
				# Compile the classifier using the configuration we want
				loaded_model.compile(optimizer=opt, loss=loss, metrics=metrics)
				
			cnn_history = loaded_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
			score = loaded_model.evaluate(X_test, y_test, verbose=0)
			print('Test Loss: ', score[0])
			print('Test Accuracy: ', score[1])

			# Kaggle user - https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
			def plot_history(history):
				loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
				val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
				acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
				val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
				
				if len(loss_list) == 0:
					print('Loss is missing in history')
					return 
				
				## As loss always exists
				epochs = range(1,len(history.history[loss_list[0]]) + 1)
				
				## Loss
				plt.figure(1)
				for l in loss_list:
					plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
				for l in val_loss_list:
					plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
				
				plt.title('Loss')
				plt.xlabel('Epochs')
				plt.ylabel('Loss')
				plt.legend()
				plt.savefig('loss-gen-{}-training_data.png'.format(sub_generation))
				plt.close()
				
				## Accuracy
				plt.figure(2)
				for l in acc_list:
					plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
				for l in val_acc_list:    
					plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

				plt.title('Accuracy')
				plt.xlabel('Epochs')
				plt.ylabel('Accuracy')
				plt.legend()
				plt.savefig('accuracy-gen-{}-training_data.png'.format(sub_generation))
				plt.close()
				
			plot_history(cnn_history)
			
			# Saving the cnn model
			if not os.path.exists("models/"):
				os.makedirs("models/")
			cnn_model_json = loaded_model.to_json()
			with open("models/cnn_model.json", "w") as json_file:
				json_file.write(cnn_model_json)
			loaded_model.save_weights("models/cnn_model.j5")
			print("Saved model to disk.")





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Generates captcha samples")
	parser.add_argument('-f', '--fonts', required=True, help='Directory containing entirely of fonts, extension .ttf')
	parser.add_argument('-g', '--generations', required=True, help='Number of generations to run for.')
	parser.add_argument('-b', '--batchsize', required=True, help='Batch size per generation created.')
	args = vars(parser.parse_args())

	train = Train(args)
	train.generate_and_train()