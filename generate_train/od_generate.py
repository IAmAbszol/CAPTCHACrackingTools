import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import sys, os
import argparse
import random
import datetime
import threading

from PIL import Image, ImageFile
from pathlib import Path
from modified_claptcha import Claptcha
from modified_image import ImageCaptcha
from modified_generator import SimpleGenerator

from utils import helper

# Fix truncation error
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Generate:

	def __init__(self, args, thread):
		# Setup - Commented optimal line as Z apparently wasn't in range?  
		#self.Characters = [chr(i) for i in range(65, 91)]
		self.Characters = helper.training_characters
		self.Sheet = pd.DataFrame(columns=['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax'])
		self.Directory = directory = args['directory'].replace("/", "") + "-{}".format(thread)
		self.Sample_size = args["size"]
		self.Fonts = ["fonts/{}".format(i) for i in os.listdir(args['fonts'])]
		self.Thread = thread
		
		# Create sub directory for image samples and csv sheet
		if not os.path.isdir(self.Directory):
			os.mkdir(self.Directory)
		else:
			# Stackoverflow solution - Clear directory contents
			for f in os.listdir(self.Directory):
				f_path = os.path.join(self.Directory, f)
				try:
					if os.path.isfile(f_path):
						os.unlink(f_path)
				except Exception as e:
					print(e)

	def generate_samples(self, set_size=1, length=4):
		start = datetime.datetime.now()
		generator = SimpleGenerator()
		for s in range(int(set_size)):
		
			if s % 100 == 0:
				print("Thread{} : Completed {}/{} - {}.".format(self.Thread, s, int(set_size), (datetime.datetime.now() - start)))
			word = ''.join([str(self.Characters[random.randrange(0, len(self.Characters))]) for i in range(random.randint(1, int(length)))])
			
			# Claptcha captcha storage
			c = Claptcha(word, "{}".format(self.Fonts[random.randint(0, (len(self.Fonts))-1)]), resample=Image.BICUBIC, noise=(random.randint(0, 4) / 10))
			text, image, offsets = c.image
			
			claptcha_offsets = list(zip(offsets[0], offsets[2]))
			for idx, c in enumerate(claptcha_offsets):
				self.Sheet = self.Sheet.append(pd.DataFrame(data=[['{}/claptcha-{}.png'.format(self.Directory,len(os.listdir(self.Directory))), 'Character', c[1], image.size[1], c[0], 0, (c[0] + c[1]), image.size[1]]], columns=['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']))
			image.save('{}/claptcha-{}.png'.format(self.Directory,len(os.listdir(self.Directory))), 'PNG')
			'''			
			# Easy captcha storage
			easy_captcha, offsets = generator.make_captcha(string=word)
			image = easy_captcha.convert()
			
			easy_offsets = [(offset[0], offset[1] - offset[0]) for offset in offsets]
			for idx, c in enumerate(easy_offsets):
				self.Sheet = self.Sheet.append(pd.DataFrame(data=[['{}/easy-{}.png'.format(self.Directory,len(os.listdir(self.Directory))), word[idx], c[1], image.size[1], c[0], 0, (c[0] + c[1]), image.size[1]]], columns=['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']))
			image.save('{}/easy-{}.png'.format(self.Directory,len(os.listdir(self.Directory))), 'PNG')
			'''
			# Captcha storage
			img_captcha = ImageCaptcha(width=40*int(length), height=70, fonts=self.Fonts)
			data, offsets = img_captcha.generate(word)
			data.seek(0)
			image = Image.open(data)
			captcha_offsets = [(offset[0], (offset[1] - offset[0])) for offset in offsets]
			for idx, c in enumerate(captcha_offsets):
				self.Sheet = self.Sheet.append(pd.DataFrame(data=[['{}/captcha-{}.png'.format(self.Directory,len(os.listdir(self.Directory))), 'Character', c[1], image.size[1], c[0], 0, (c[0] + c[1]), image.size[1]]], columns=['filename', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']))
			image.save('{}/captcha-{}.png'.format(self.Directory,len(os.listdir(self.Directory))), 'PNG')
			
		self.Sheet = self.Sheet.reset_index()
		self.Sheet = self.Sheet.drop(['index'], axis=1)
		self.Sheet.to_csv("{}/sheet.csv".format(self.Directory), sep=',', header=True)


def enact_threads(args, thread):
	# Generate the samples
	generate = Generate(args, thread)
	generate.generate_samples(set_size=args["size"], length=args["length"])
				
			
			
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generates captcha samples")
	parser.add_argument('-s', '--size', required=True, help='Selected size of sample base')
	parser.add_argument('-l', '--length', required=True, help='Selected length of word created')
	parser.add_argument('-d', '--directory', required=True, help='Designated save directory')
	parser.add_argument('-f', '--fonts', required=True, help='Directory containing entirely of fonts, extension .ttf')
	parser.add_argument('-t', '--threads', required=False, help='Number of threads to enact, highly suggested! Default is 1.')
	args = vars(parser.parse_args())
	threads = int(args['threads']) if args['threads'] is not None and int(args['threads']) > 0 else 1
	thread_list = []
	for t in range(threads):
		thread = threading.Thread(target=enact_threads, args=(args, t,))
		thread.start()
		thread_list.append(thread)
	for t in thread_list:
		t.join()
	# Merge all samples if threads > 1 and not None
	if threads > 1:
		master_sheet = None
		for t in range(threads):
			directory = args['directory'].replace("/", "") + "-{}/sheet.csv".format(t)
			if master_sheet is None:
				master_sheet = pd.read_csv(directory, sep=',')
				master_sheet = master_sheet.reset_index()
				master_sheet = master_sheet.drop(['index'], axis=1)
				continue
			sheet = pd.read_csv(directory, sep=',')
			sheet = sheet.reset_index()
			sheet = sheet.drop(['index'], axis=1)
			master_sheet = pd.concat([master_sheet, sheet], ignore_index=True)
		
		master_sheet = master_sheet.drop(['Unnamed: 0'], axis=1)
		master_sheet.to_csv("sheet.csv", sep=',', header=True, index=False)
		
		my_sheet = Path("sheet.csv")
		if my_sheet.is_file():
			with open("sheet.csv", "a") as f:
				master_sheet.to_csv(f, sep=',', header=False, index=False)
		else:
			master_sheet.to_csv("sheet.csv", sep=',', header=True, index=False)

			
