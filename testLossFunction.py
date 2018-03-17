from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from tomsNet.network import Network, loss_l2_norm, getFilenameQueuesFromCSVFile, csv_inputs

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

'''
    image_reader = tf.WholeFileReader()
    
    filename_queue1 = tf.train.string_input_producer(["./testImages/00609.jpg"])
    depth_filename = tf.train.string_input_producer(["./testImages/00609.png"])

    _, image_file1 = image_reader.read(filename_queue1)
    image1 = tf.image.decode_jpeg(image_file1, channels=3)
    images1 = tf.image.resize_images([image1], imageSize)

    _, depth = image_reader.read(depth_filename)
    depth = tf.image.decode_png(depth, channels=3)
    depth = tf.image.resize_images([depth], depthImageSize)
    depth = tf.slice(depth, [0,0,0, 0], [-1,-1,-1,1])
    depth = tf.divide(depth, 100.0)
'''

'''
    f, g = getFilenameQueuesFromCSVFile( 'test.csv' )
    images1, depth, invalid_depths, filenames = csv_inputs(f, g, 1, imageSize, depthImageSize)
'''

def evaluate2(input_network):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    original_size = (480, 640)
    
    print('FLAGS.eval_dir')
    print(FLAGS.eval_data)

    csvFilePath = 'testingtest.csv'
    imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
    depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)

 
    image_filename, depth_filename = getFilenameQueuesFromCSVFile( csvFilePath )   
    images1, depth, invalid_depths, filenames = csv_inputs( image_filename, depth_filename , 1, imageSize, depthImageSize)


    logits1 = input_network.getInference(images1)



    print("printing the shape")
    print("printing the shape" + str(logits1.shape))
    print("printing the shape" + str(depth.shape))
    
    #total_loss = loss_l2_norm(logits1, depth, None)
    logits1 = tf.image.resize_images(logits1, (10, 10))
    depth   = tf.image.resize_images(depth  , (10, 10))
 
    logits_flat = tf.contrib.layers.flatten(logits1)
    depths_flat = tf.contrib.layers.flatten(depth)
    d = tf.subtract(logits_flat, depths_flat)
    d_test = d
    d_test = tf.abs(d_test)
    d_test = tf.reduce_sum(d_test)
    total_loss = tf.nn.l2_loss(d)

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

	return_values, images2_out, outputImage, logits_flat_out, depths_flat_out, d_out = sess.run([total_loss, depth, logits1,  logits_flat, depths_flat, d_test])
	print('logits1_out[0]')
	print(logits_flat_out[0])
	print('images2_out[0]')
	print(depths_flat_out[0])
	ar1 = []
	ar2 = []
	stri = "["
	for v in logits_flat_out[0]:
		stri = stri + ',' + str(v)
		ar1.append(v)
	stri = stri + ']'
	print(stri)
	stri = "["
	for v in depths_flat_out[0]:
		stri = stri + ',' + str(v)
		ar2.append(v)

	total = 0
	stri = "["
	for i in range(len(ar1)):
		#print( str(ar1[i]) + " - " + str(ar2[i]) + " = " + str(ar1[i] - ar2[i]) )
		total = total + abs(ar1[i] - ar2[i])
		stri = stri + ',' + str(abs(ar1[i] - ar2[i]))
	print('total')
	print(total)
	print('d_out')
	print(d_out)
	stri = stri + ']'
	print(stri)
	pred2 = images2_out
	formatted2 = ((pred2[0,:,:,0]) * 255 / np.max(pred2[0,:,:,0])).astype('uint8')
	pred = outputImage
	formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
	#print("pred")
	#print(pred[0])
	img = Image.fromarray(formatted)
	img.save("./output.png")
	img = Image.fromarray(formatted2)
	img.save("./output2.png")
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
