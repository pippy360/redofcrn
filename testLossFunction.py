from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from tomsNet.network import Network, loss_l2_norm

from network2.network import theNetwork 
import models 
import argparse

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--eval_dir', type=str, default='./eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='data/nyu_datasets/00000.jpg',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='./train_checkpoint',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=120,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=1000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')

parser.add_argument('--n1', type=bool, default=False,
                    help='Whether to run eval only once.')

batch_size = 1

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 8


def evaluate1(input_network):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
	#with tf.variable_scope("InnerScope" , reuse=tf.AUTO_REUSE):
	    # Get images and labels for CIFAR-10.
	    original_size = (480, 640)
	    
	    imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
	    depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
	    print('FLAGS.eval_dir')
	    print(FLAGS.eval_data)

	    image_reader = tf.WholeFileReader()
	    
	    filename_queue1 = tf.train.string_input_producer(["./0.jpg"])

	    _, image_file1 = image_reader.read(filename_queue1)

	    image1 = tf.image.decode_jpeg(image_file1, channels=3)

	    images1 = tf.image.resize_images([image1], imageSize)
	    logits1 = input_network.getInference(images1)


	    filename_queue2 = tf.train.string_input_producer(["./0.jpg"])

	    _, image_file2 = image_reader.read(filename_queue2)

	    image2 = tf.image.decode_jpeg(image_file2, channels=3)

	    images2 = tf.image.resize_images([image2], imageSize)
	    logits2 = input_network.getInference(images2)
	    
	    total_loss = loss_l2_norm(logits1, logits2, None)

	    summary_op = tf.summary.merge_all()

	    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
	    with tf.Session() as sess:
	      global_step = input_network.restore(sess)
	      # Start the queue runners.
	      coord = tf.train.Coordinator()
	      try:
		threads = []
		for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
		  threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
						   start=True))

		return_values, filename1, filename2 = sess.run([total_loss, image_file1, image_file2])
		print('return_values')      
		print(return_values)      

		summary = tf.Summary()
		summary.ParseFromString(sess.run(summary_op))
		summary_writer.add_summary(summary, global_step)
	      except Exception as e:  # pylint: disable=broad-except
		coord.request_stop(e)

	      coord.request_stop()
	      coord.join(threads, stop_grace_period_secs=10)

def evaluate2(input_network):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    original_size = (480, 640)
    
    imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
    depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
    print('FLAGS.eval_dir')
    print(FLAGS.eval_data)

    image_reader = tf.WholeFileReader()
    
    filename_queue1 = tf.train.string_input_producer(["./testImages/00609.jpg"])

    _, image_file1 = image_reader.read(filename_queue1)

    image1 = tf.image.decode_jpeg(image_file1, channels=3)

    images1 = tf.image.resize_images([image1], imageSize)
    logits1 = input_network.getInference(images1)


    filename_queue2 = tf.train.string_input_producer(["./output.png"])

    _, image_file2 = image_reader.read(filename_queue2)

    image2 = tf.image.decode_png(image_file2, channels=3)

    images2 = tf.image.resize_images([image2], depthImageSize)
    images2 = tf.slice(images2, [0,0,0, 0], [-1,-1,-1,1])
    print("printing the shape")
    print("printing the shape" + str(logits1.shape))
    print("printing the shape" + str(images2.shape))
    total_loss = loss_l2_norm(logits1, images2, None)

    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    with tf.Session() as sess:
      global_step = input_network.restore(sess, True)
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
	threads = []
	for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
	  threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
					   start=True))

	return_values, filename1, filename2, outputImage = sess.run([total_loss, image_file1, image_file2, logits1])
	pred = outputImage
	formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
	#print("pred")
	#print(pred[0])
	img = Image.fromarray(formatted)
	img.save("./output.png")
	#return_values = sess.run([ images2 ])
	print('return_values')      
	print(return_values)      
	summary = tf.Summary()
	summary.ParseFromString(sess.run(summary_op))
	summary_writer.add_summary(summary, global_step)
      except Exception as e:  # pylint: disable=broad-except
	coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)




def main(argv=None):  # pylint: disable=unused-argument
  input_network = theNetwork()
  evaluate2(input_network)


if __name__ == '__main__':
  FLAGS = parser.parse_args()
tf.app.run()
