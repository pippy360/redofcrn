from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import network
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

def eval_once(saver, summary_writer, logits, summary_op):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    chpt_dir = FLAGS.checkpoint_dir

    if FLAGS.n1:
      chpt_dir = './checkpoint_dir2'
    
    ckpt = tf.train.get_checkpoint_state(chpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('global_step')
      print(global_step)
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      return_values = sess.run(logits)
      pred = return_values[0]
      print('pred')
      formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
      img = Image.fromarray(formatted)
      img.save("./output.jpg")
      #save it....
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)



def evaluate(input_network):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    original_size = (480, 640)
    
    imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
    depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
    print('FLAGS.eval_dir')
    print(FLAGS.eval_data)
    
    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    filename_queue = tf.train.string_input_producer(["./0.jpg"])

    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file, channels=3)

    images = tf.image.resize_images([image], imageSize)
    logits = input_network.inference()

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    eval_once(saver, summary_writer, [logits], summary_op)


def main(argv=None):  # pylint: disable=unused-argument
  input_network = network
  evaluate(input_network)


if __name__ == '__main__':
  FLAGS = parser.parse_args()
tf.app.run()
