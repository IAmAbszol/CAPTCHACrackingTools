'''
	Author : Kyle Darling <kdarling95@yahoo.com>
	Description : This file is a designed test case to take in
	the trained model via TensorFlow's Object Detection model.
	
	Code adapted from the following Github tutorial:
	https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
'''

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import keras
import pandas as pd
import seaborn as sns
import datetime
import random

# CAPTCHA resources
from captcha.image import ImageCaptcha
from captcha.generator import SimpleGenerator
from claptcha import Claptcha

# TensorFlow's utils
from tf_utils import label_map_util
from tf_utils import visualization_utils as vis_util
from keras.models import model_from_json

from sklearn import preprocessing
from PIL import Image

from utils import helper

from matplotlib import pyplot as plt

# Testing Parameters
claptcha_data = []
easy_captcha_data = []
captcha_data = []
training_length = 200
character_length = 6

# Create CAPTCHA images from the CLAPTCHA module
characters = helper.training_characters
# Uncomment and recomment below line to evaluate multiple fonts
#fonts = ["../utils/fonts/{}".format(i) for i in os.listdir("../utils/fonts/") if ".ttf" in i.lower() or ".otf" in i.lower()]
fonts=["../utils/fonts/arial.ttf"]
generator = SimpleGenerator()

# Loading segmentation model
MODEL_NAME = ""
CWD_PATH = os.getcwd() 

if len(sys.argv) > 1:
	MODEL_NAME = sys.argv[1]
else:
	MODEL_NAME = "inference_graph"
	
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Number of unique characters
NUM_CLASSES = len(characters)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
	
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load Character Recognition Model
model = "{}.json".format("models/cnn_model")
weights = "{}.j5".format("models/cnn_model")

json_file = open(model, "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights)

loaded_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=.001), metrics=['accuracy'])

label_encoder = preprocessing.LabelEncoder()
y_labeled = label_encoder.fit_transform(characters)

# Begin Clock
start_time = datetime.datetime.now()

print("Generating CLAPTCHA images.")
for train in range(training_length):
	
	if train % 10 == 0:
		print("Completed {}/{} - {}.".format(train, training_length, (datetime.datetime.now() - start_time)))
		
	word = ''.join([characters[random.randrange(0, len(characters))] for i in range(character_length)])
	try:
		claptcha = Claptcha(word, fonts[random.randint(0, (len(fonts))-1)], resample=Image.BICUBIC, noise=(random.randint(0, 6))/10)
		text, image = claptcha.image
		claptcha_data.append((text, image))
		
		image = ImageCaptcha(fonts=fonts)
		data = image.generate(word)
		data.seek(0)
		img = Image.open(data)
		captcha_data.append((word, img))
		
	except Exception as e:
		print(e)
		exit(0)

def tod_split(image):
	guess = []
	# Convert image to cv2 (numpy array)
	np_image = np.array(image)
	image_expanded = np.expand_dims(np_image, axis=0)
	
	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
	[detection_boxes, detection_scores, detection_classes, num_detections],
	feed_dict={image_tensor: image_expanded})
	
	# Retrieve characters
	# Draw the results of the detection (aka 'visulaize the results')
	img, coordinates = vis_util.visualize_boxes_and_labels_on_image_array(
		np_image,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.80)
	
	# Sort list from least to greatest X
	coordinates.sort(key=lambda tp: tp[0])
	
	for xmin, ymin, xmax, ymax in coordinates:
		s = image.crop((xmin, ymin, xmax, ymax))
		#s.save('{}.png'.format(xmin))
		sub_image = s
		sub_image = sub_image.convert('L')
		np_image = np.array(sub_image.resize((28, 34), Image.ANTIALIAS))
		np_image = np_image.reshape(1, 28, 34, 1)
		np_image = np_image.astype('float32')
		np_image /= 255
		
		prediction = loaded_model.predict(np_image)
		prediction = np.where(prediction > 0.5, 1, 0)
		guess.append(str(label_encoder.inverse_transform([np.argmax(prediction)])[0]))
	return guess
	
def peak_split(image):
	guess = []
	characters = helper.peak_segmentation(image)
	if characters == -1:
		return guess
	for character in characters:
		sub_image = image.crop((character[0], 0, character[1], image.size[1]))
		sub_image = sub_image.convert('L')
		np_image = np.array(sub_image.resize((28, 34), Image.ANTIALIAS))
		np_image = np_image.reshape(1, 28, 34, 1)
		np_image = np_image.astype('float32')
		np_image /= 255
		
		prediction = loaded_model.predict(np_image)
		prediction = np.where(prediction > 0.5, 1, 0)
		guess.append(str(label_encoder.inverse_transform([np.argmax(prediction)])[0]))
	return guess
		
names = ['Claptcha', 'Captcha']
for n_idx, data in enumerate([claptcha_data, captcha_data]):
#for idx, data in enumerate([easy_captcha_data]):

	# Globals
	cracked_letters = 0
	cracked_captchas = 0
	total = 0
	bad_guess = 0
	
	evaluaters = [(tod_split, "ToD Segmentation", 0,0,0,0),(peak_split, "Peak Segmentation", 0,0,0,0)]
	character_data = [(0,0) for i in range(len(helper.training_characters))]
	
	# Begin evaluation
	for train in range(training_length):
		
		text, image = data[train]
		
		for idx, method in enumerate(evaluaters):
		
			fn_eval = method[0]
			cracked_letters = method[2]
			cracked_captchas = method[3]
			total = method[4]
			bad_guess = method[5]
			
			guess = fn_eval(image)
			
			if len(guess) != len(text):
				bad_guess += 1
				# Repack
				evaluaters[idx] = (method[0], method[1], cracked_letters, cracked_captchas, total, bad_guess)
				#print(coordinates, names[idx])
				continue
			
			# Evaluate correctness of characters
			if idx == 0:
				for t_idx, char in enumerate(text):
					char_idx = helper.training_characters.index(char)
					tp, fp = character_data[char_idx]
					tp += 1 if char == guess[t_idx] else 0
					fp += 1 if char != guess[t_idx] else 0
					character_data[char_idx] = (tp, fp)
			
			cracked_captchas += 1 if sum(1 for idx, i in enumerate(list(text)) if str(i) == str(guess[idx])) == len(guess) else 0
			cracked_letters += sum(1 for idx, i in enumerate(list(text)) if str(i) == str(guess[idx]))

			# Repack
			evaluaters[idx] = (method[0], method[1], cracked_letters, cracked_captchas, total, bad_guess)
			
		
	print("{}: Completed in {}".format(names[n_idx], datetime.datetime.now() - start_time))
	for idx, method in enumerate(evaluaters):
		print("Method {}".format(method[1]))
		print("Cracked Characters: {}/{} - {}%".format(method[2], (training_length) * character_length, (method[2]/((training_length)*character_length))*100))
		print("Cracked CAPTCHAs: {}/{} - {}%".format(method[3], (training_length), (method[3]/(training_length))*100))
		print("Cracks per second: {}".format(training_length/(datetime.datetime.now() - start_time).total_seconds()))
		print("Bad Segmentations: {}/{} - {}%".format(method[5], training_length, (method[5]/training_length)*100))
		print("-"*50)
		
		if idx == 0:
			ind = np.arange(len(character_data))
			tp_bar = plt.bar(ind, [tp[0] for tp in character_data], .35)
			fp_bar = plt.bar(ind, [fp[1] for fp in character_data], .35)
			
			plt.title(names[n_idx])
			plt.xlabel('Characters')
			plt.ylabel('Frequencies')
			plt.xticks(ind, helper.training_characters)
			plt.legend((tp_bar[0], fp_bar[0]), ('TP', 'FP'))
			
			plt.savefig('{}-{}.png'.format(names[n_idx], training_length))
			plt.close()

