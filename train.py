import tensorflow as tf
import os
from PIL import Image
import numpy as np
import math
import time
import datetime

from tomsNet.network import Network
from tomsNet.network import csv_inputs, loss_l2_norm

from network2.network import theNetwork



CHECKPOINT_DIR = './train_checkpoint'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 8

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
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  #for var in tf.trainable_variables():
  #  tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  #for grad, var in grads:
  #  if grad is not None:
  #    tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def runIt(inputNetwork):
    with tf.Graph().as_default():
	global_step = tf.train.get_or_create_global_step()
        imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
        depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
        filename_queue = tf.train.string_input_producer([TRAIN_FILE], shuffle=False)
        images, depths, invalid_depths, filenames = csv_inputs(filename_queue, BATCH_SIZE, imageSize=imageSize, depthImageSize=depthImageSize)
        logits = inputNetwork.getInference(images)
        tf.summary.image('input_images', logits, max_outputs=3)
        #loss_op = loss_scale_invariant_l2_norm(logits, depths, invalid_depths)
        loss_op = loss_l2_norm(logits, depths, invalid_depths)
	train_op = train(loss_op, global_step, batch_size=1)
        init = tf.global_variables_initializer()

	class _LoggerHook(tf.train.SessionRunHook):
	  """Logs loss and runtime."""

	  def begin(self):
	    self._step = -1
	    self._start_time = time.time()

	  def before_run(self, run_context):
	    self._step += 1
	    return tf.train.SessionRunArgs([loss_op, logits])  # Asks for loss value.

	  def after_run(self, run_context, run_values):
            log_frequency = 1
	    if self._step % log_frequency == 0:
	      current_time = time.time()
	      duration = current_time - self._start_time
	      self._start_time = current_time

	      loss_value = run_values.results[0]
              #filename = run_values.results[1]
              batch_size = 1
	      examples_per_sec = log_frequency * batch_size / duration
	      sec_per_batch = float(duration / log_frequency)
              output_images = run_values.results[1]
              #TODO: FIXME: tf.summary.image('output_images', output_images, max_outputs=3)
	      
              format_str = (': step %d, loss = %.2f (%.1f examples/sec; %.3f '
			    'sec/batch)')
	      print (format_str % (self._step, loss_value,
				   examples_per_sec, sec_per_batch))

	with tf.train.MonitoredTrainingSession(
	    checkpoint_dir=inputNetwork.getCheckpointDir(),
	    hooks=[tf.train.StopAtStepHook(last_step=100000),
		   tf.train.NanTensorHook(loss_op),
                   _LoggerHook()],
	    ) as mon_sess:
                print 'running...'
                while not mon_sess.should_stop():
                    print 'step...'
		    mon_sess.run(train_op)



def main(argv=None):  # pylint: disable=unused-argument
	if not os.path.exists(CHECKPOINT_DIR):
		os.makedirs(CHECKPOINT_DIR)
	inputNetwork = theNetwork()
	runIt(inputNetwork)

main()
