# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import datetime
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains 
from urllib.request import urlretrieve

# TensorFlow's utils
from tf_utils import label_map_util
from tf_utils import visualization_utils as vis_util
from keras.models import model_from_json

from sklearn import preprocessing
from PIL import Image
from matplotlib import pyplot as plt

from utils import helper

# Load local data
dataframe = pd.read_csv('captchas.csv', sep=',')
labels = [l for l in dataframe['label']]
images = [Image.open(f) for f in dataframe['image']]
data = list(zip(images, labels))

# Loading segmentation model
MODEL_NAME = ""
CWD_PATH = os.getcwd() 

MODEL_NAME = "inference_graph"
	
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Number of unique characters
NUM_CLASSES = len(helper.training_characters)

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
y_labeled = label_encoder.fit_transform(helper.training_characters)

# Begin Clock
start_time = datetime.datetime.now()

# Globals
cracked_letters = 0
cracked_captchas = 0
total = len(data)
bad_guess = 0

character_data = [(0,0) for i in range(len(helper.training_characters))]

for idx, (image, label) in enumerate(data):

	guess = []

	# Convert image to cv2 (numpy array)
	np_image = np.array(image)
	np_image = np_image[...,:3]
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

	coordinates = reduced_x_axis = helper.reconstruct(coordinates, 6)
	if coordinates == -1:
		bad_guess +=1 
		continue
	
	for xmin, ymin, xmax, ymax in coordinates:
		sub_image = image.crop((xmin, ymin, xmax, ymax))
		sub_image = sub_image.convert('L')
		
		# Peak segment them
		revised_ind = helper.peak_segmentation(sub_image)
		np_image = np.array(sub_image.resize((28, 34), Image.ANTIALIAS))
		np_image = np_image.reshape(1, 28, 34, 1)
		np_image = np_image.astype('float32')
		np_image /= 255
		
		prediction = loaded_model.predict(np_image)
		prediction = np.where(prediction > 0.5, 1, 0)
		guess.append(str(label_encoder.inverse_transform([np.argmax(prediction)])[0]))
			
	# Prediction
	prediction = ''.join(guess)
	
	# Evaluate
	if len(prediction) != len(label):
		bad_guess += 1
		continue

	for t_idx, char in enumerate(label):
		char_idx = helper.training_characters.index(char)
		tp, fp = character_data[char_idx]
		tp += 1 if char == guess[t_idx] else 0
		fp += 1 if char != guess[t_idx] else 0
		character_data[char_idx] = (tp, fp)
	
	cracked_captchas += 1 if sum(1 for idx, i in enumerate(list(label)) if str(i) == str(guess[idx])) == len(guess) else 0
	cracked_letters += sum(1 for idx, i in enumerate(list(label)) if str(i) == str(guess[idx]))
	
	plt.imshow(image)
	plt.text(0, 10, prediction)
	plt.savefig('{}.png'.format(idx))
	plt.close()
	
print("Cracked Characters: {}/{} - {}%".format(cracked_letters, (total) * len(label), (cracked_letters/((total)*len(label)))*100))
print("Cracked CAPTCHAs: {}/{} - {}%".format(cracked_captchas, (total), (cracked_captchas/(total))*100))
print("Cracks per second: {}".format(total/(datetime.datetime.now() - start_time).total_seconds()))
print("Bad Segmentations: {}/{} - {}%".format(bad_guess, total, (bad_guess/total)*100))

ind = np.arange(len(character_data))
tp_bar = plt.bar(ind, [tp[0] for tp in character_data], .35)
fp_bar = plt.bar(ind, [fp[1] for fp in character_data], .35)

plt.title('Local Benchmark')
plt.xlabel('Characters')
plt.ylabel('Frequencies')
plt.xticks(ind, helper.training_characters)
plt.legend((tp_bar[0], fp_bar[0]), ('TP', 'FP'))

plt.savefig('local_benchmark.png')
plt.close()