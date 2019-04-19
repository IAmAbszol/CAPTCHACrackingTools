import sys
import numpy as np

from sklearn import preprocessing
from PIL import Image

#training_characters = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '@', '#', '$','%', '^', '&', '*']
#training_characters = training_characters[:24]
training_characters = ['0','1','2','3','4','5','6','7','8','9']

# Optimization could be done and reduce to ~O(n) with varying checks
def peak_segmentation(image, threshold=5):
	
	# Segment the image through use of vertical bars
	np_image = np.array(image.convert('L'))
	np_image = np_image.astype('float32')
	np_image /= 255
	
	segment_columns = []
	
	# Grab the column, perform vertical analysis via the rows
	for col in range(len(np_image[0])):
		sums = 0
		for row in range(len(np_image)):
			sums += np_image[row][col]
		segment_columns.append(len(np_image[0]) - sums)
	
	# Preprocess the y-axis, it'll distintively show character cloud clusters.
	X = np.array([i for i in range(len(np_image[0]))])
	y = preprocessing.scale(np.array(segment_columns))
	
	# Remove borders of the known starting and ending characters
	for i in range(len(y)):
		if y[i] <= 0:
			y[i] = 0
		else:
			for j in range(len(y) - 1, 0, -1):
				if y[j] <= 0:
					y[j] = 0
				else:
					break
			break

	# Sum all values less than 0 and gain an average.
	# Then remove all values less than the average (avg --> -inf)
	avg = 1
	#total = 0
	#for i in y:
	#	if i <= 0:
	#		avg += i
	#		total += 1
	#avg /= total
	y = [i if i >= avg else 0 for i in y]

	# Finally iterate through the y-axis
	# Every value > 0 will be our character, hence
	# we iterate until we hit ~3 0's (Arbitrary)
	# * Don't critique, sleep deprivation is winning
	character_indices = []
	start = 0
	reading = False
	count = 0
	for index, i in enumerate(y):
		if i != 0:
			count = 0
			if not reading:
				reading = True
				start = index
		if i == 0:
			count += 1
			if reading and count >= threshold:
				reading = False
				character_indices.append((start, index))
				count = 0

	#print(character_indices)
	#from matplotlib import pyplot as plt
	#plt.bar(X, y)
	#plt.show()
	
	return character_indices
	

# Have reconstruct instead look at the end and starting indicies of two respective
# points whom are neighbors to one another. Find the smallest gap, combine, then re-iterate over
# till desired is reached
def combine_ends(character_indices, correct_size):
	while len(character_indices) > correct_size:
		idx, minimum = 0, sys.maxsize
		for i in range(1, len(character_indices)):
			if character_indices[i][0] - character_indices[i - 1][1] < minimum:
				minimum = character_indices[i][0] - character_indices[i - 1][1]
				idx = i
		# Remove and get the character_indices tuple
		val = character_indices.pop(idx)
		character_indices[idx - 1] = (character_indices[idx - 1][0], val[1])
	return character_indices
	
def reconstruct(character_indices, correct_size):
	"""
		Compresses smallest to highest indicies
		:params character_indices: Tupled list format of (xmin, ymin, xmax, ymax).
		:params coorect_size: Dictates the size of the list that it should be after reduction.
		:return: Returns reduced character indicies set that was passed through, same format.
	"""
	while len(character_indices) > correct_size:	

		# Patch fix, evaluate index widths
		min_width = 100 # Change if need be
		idx = 0
		for index, character in enumerate(character_indices):
			width = character[2] - character[0]
			if width < min_width:
				min_width = width
				idx = index

		# Grab values between index
		if 0 < idx < len(character_indices) - 1: 
			lhs = character_indices[idx-1][2] - character_indices[idx-1][0]
			rhs = character_indices[idx+1][2] - character_indices[idx+1][0]
			# RHS is the culprit, repair. If equal, favor rhs
			if lhs >= rhs:
				rhs = character_indices[idx+1][2]
				character_indices[idx] = (character_indices[idx][0], character_indices[idx][1], rhs, character_indices[idx][3])
				del character_indices[idx+1]
			elif lhs < rhs:
				lhs = character_indices[idx-1][0]
				character_indices[idx] = (lhs, character_indices[idx][1], character_indices[idx][2], character_indices[idx][3])
				del character_indices[idx-1]
		
		# Our problem lays either at the start or end. Same process
		if idx == 0:
			character_indices[idx] = (character_indices[idx][0], 
										min(character_indices[idx][1], character_indices[idx+1][1]), 
										character_indices[idx+1][2], 
										max(character_indices[idx][3], character_indices[idx+1][3]))
			del character_indices[idx+1]
		elif idx == len(character_indices) - 1:
			character_indices[idx] = (character_indices[idx-1][0], 
										min(character_indices[idx-1][1], character_indices[idx][1]), 
										character_indices[idx][2],
										max(character_indices[idx][3], character_indices[idx-1][3]))
			del character_indices[idx-1]

	if len(character_indices) < correct_size:
		return -1

	return character_indices


def combine_image(sub_one, sub_two):

	total_width = sub_one.size[0] + sub_two.size[0]
	max_height = sub_one.size[1]
	'''
	files = ['2.png', '3.png']
	images = [Image.open(i) for i in files]
	widths, heights = zip(*(i.size for i in images))

	total_width = sum(widths)
	max_height = max(heights)
	'''
	new_im = Image.new('RGB', (total_width, max_height))

	x_offset = 0
	'''
	for im in images:
	  new_im.paste(im, (x_offset,0))
	  x_offset += im.size[0]
	'''
	new_im.paste(sub_one, (x_offset,0))
	x_offset += sub_one.size[0]
	new_im.paste(sub_two, (x_offset,0))
	new_im.save('image.png')

	return new_im
