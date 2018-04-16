import tensorflow as tf
import os
from PIL import Image
import numpy as np
import math
import time
import datetime

from tomsNet.network import Network
from tomsNet.network import getFilenameQueuesFromCSVFile, csv_inputs, loss_l2_norm

from network2.network import theNetwork



CHECKPOINT_DIR = './train_checkpoint'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 30
INITIAL_LEARNING_RATE = 0.000000001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op



def train(total_loss, global_step, batch_size):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train_op = optimizer.minimize(total_loss)
  return train_op


def runIt(inputNetwork):
    with tf.Graph().as_default():
	global_step = tf.train.create_global_step()
        imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
        depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
	filename, depth_filename = getFilenameQueuesFromCSVFile( TRAIN_FILE )
        images, depths, invalid_depths, filenames = csv_inputs(filename, depth_filename, BATCH_SIZE, imageSize=imageSize, depthImageSize=depthImageSize)
        logits = inputNetwork.getInference(images)
        tf.summary.image('input_images', logits, max_outputs=3)
        #loss_op = loss_scale_invariant_l2_norm(logits, depths, invalid_depths)
        loss_op = loss_l2_norm(logits, depths, invalid_depths)
	train_op = train(loss_op, global_step, batch_size=1)
        init = tf.global_variables_initializer()
	with tf.Session() as sess:
		network.restore(sess, True)
		print("init run")
		for i_iter in range(1000):
			_, loss = sess.run((train_op, loss_op))
			print("loss " + str(loss))


def main(argv=None):  # pylint: disable=unused-argument
	if not os.path.exists(CHECKPOINT_DIR):
		os.makedirs(CHECKPOINT_DIR)
	inputNetwork = theNetwork()
	runIt(inputNetwork)

main()
