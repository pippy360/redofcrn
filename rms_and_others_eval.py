from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf


from tomsNet.network import Network
from tomsNet.network import csv_inputs

from network2.network import theNetwork

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--eval_dir', type=str, default='./eval',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test2.csv',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='./train_checkpoint',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=120,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=1000,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')

batch_size = 1

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 8

def eval_once(summary_writer, loss_ops, summary_op, network_input):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    global_step = network_input.restore(sess)
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / batch_size))
      abs_relative_diff_average = 0.0
      abs_relative_diff_squared_average = 0.0
      rmse_average = 0.0
      total_sample_count = num_iter * batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        return_values = sess.run(loss_ops)
        abs_relative_diff_average += float(return_values[0])/float(num_iter)
        abs_relative_diff_squared_average += float(return_values[1])/float(num_iter)
	rmse_average += float(return_values[2])/float(num_iter)

        print('%s: loss_average = abs: %.3f, abs_squared: %.3f, rmse: %.3f' % (datetime.now(), 
		return_values[0], return_values[1], return_values[2]))
        
        step += 1

      print('%s: loss_average = abs: %.3f, abs_squared: %.3f, rmse: %.3f' % (datetime.now(), 
		abs_relative_diff_average, abs_relative_diff_squared_average, rmse_average))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='eval_abs_relative_diff_average', simple_value=abs_relative_diff_average)
      summary.value.add(tag='eval_abs_relative_diff_squared_average', simple_value=abs_relative_diff_squared_average)
      summary.value.add(tag='eval_rmse_average', simple_value=rmse_average)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)



def evaluate(network_input):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    original_size = (480, 640)
    
    imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
    depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
    filename_queue = tf.train.string_input_producer([FLAGS.eval_data], shuffle=True)
    images, depths, invalid_depth, filenames = csv_inputs(filename_queue, batch_size, imageSize=imageSize, depthImageSize=depthImageSize)
    logits = network_input.getInference(images)

    predict = logits
    target = depths
 
    predict_resize = tf.image.resize_images(predict, original_size)
    target_resize  = tf.image.resize_images(target, original_size)
    
    predict_flat = tf.reshape(predict_resize, [-1, 480*640])
    target_flat  = tf.reshape(target_resize, [-1, 480*640])

    print(predict_flat)
    print(target_flat)

    #TODO: you need to actually confirm this works
    abs_relative_diff_op = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(predict_flat, target_flat)), target_flat))

    #TODO: you need to actually confirm this works
    abs_relative_diff_squared_op = tf.reduce_mean(tf.divide(tf.abs(tf.squared_difference(predict_flat, target_flat)), target_flat))
    
    rmse_op = tf.sqrt(tf.reduce_mean(tf.squared_difference(predict_flat, target_flat))) 

    # Restore the moving average version of the learned variables for eval.

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    eval_once(summary_writer, [abs_relative_diff_op, abs_relative_diff_squared_op, rmse_op], summary_op, network_input)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  input_network = theNetwork()
  evaluate(input_network)

if __name__ == '__main__':
  FLAGS = parser.parse_args()
tf.app.run()
