import numpy as np
import pandas as pd
import sys, os
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from PIL import Image

class Reload:

	def __init__(self, args):
		self.Data = np.array(pd.read_csv(args["file"], converters={"x": lambda z: z.strip("[]").split(", "), "w": lambda y: y.strip("[]").split(", "), "h": lambda x: x.strip("[]").split(", ")}, sep=','))
		for row in self.Data:
			image = Image.open(row[1])
			
			if not image is None:
				fig, ax = plt.subplots(1)
				plt.imshow(image)
				for index, p in enumerate(row[5]):
					rect = patches.Rectangle((int(p), int(row[6])), int(row[4][index]), int(row[3][index]), linewidth=1, edgecolor='r', facecolor='none')
					ax.add_patch(rect)
				plt.show()
			image.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generates captcha samples")
	parser.add_argument('-f', '--file', required=True, help='Location for sheet.csv')
	args = vars(parser.parse_args())

	reload = Reload(args)
