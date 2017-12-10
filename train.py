import tensorflow as tf
import network
import os
from PIL import Image
import numpy as np

logsDir = './logs/'

def main(image_path):
	if not os.path.exists(logsDir):
		os.makedirs(logsDir)

	with tf.Session() as sess:
		# Default input size
		height = 228
		width = 304
		channels = 3
		batch_size = 1

		# Read image
		img = Image.open(image_path)
		img = img.resize([width,height], Image.ANTIALIAS)
		img = np.array(img).astype('float32')
		img = np.expand_dims(np.asarray(img), axis = 0)

		# Create a placeholder for the input image
		input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

		logits = network.inference(input_node)
		file_writer = tf.summary.FileWriter(logsDir)
		file_writer.add_graph(sess.graph)

main('./00001_org.png')

