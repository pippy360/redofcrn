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
    num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.scalar_summary('learning_rate', lr)
    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for var in tf.trainable_variables():
        print(var.op.name)
        tf.histogram_summary(var.op.name, var)
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def runIt(inputNetwork):
    with tf.Graph().as_default():
        imageSize = (IMAGE_HEIGHT, IMAGE_WIDTH)
        depthImageSize = (TARGET_HEIGHT, TARGET_WIDTH)
	filename, depth_filename = getFilenameQueuesFromCSVFile( TRAIN_FILE )
        images, depths, invalid_depths, filenames = csv_inputs(filename, depth_filename, BATCH_SIZE, imageSize=imageSize, depthImageSize=depthImageSize)
        logits = inputNetwork.getInference(images)
        tf.summary.image('input_images', logits, max_outputs=3)
        #loss_op = loss_scale_invariant_l2_norm(logits, depths, invalid_depths)
        loss_op = loss_l2_norm(logits, depths, invalid_depths)

  	optimizer = tf.train.GradientDescentOptimizer(0.0000000009)
  	train_op = optimizer.minimize(loss_op)

        init = tf.global_variables_initializer()
	with tf.Session() as sess:
		global_step = inputNetwork.restore(sess, True)
		
		coord = tf.train.Coordinator()
		try:
			threads = []
			for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
			if global_step is None:
				sess.run([init])
				
			print("init run")
			for i in range(100000):
				if not i %10 == 0:
					sess.run([train_op])
				else:
					_, loss, outputDepth, testDepth, image = sess.run([train_op, loss_op, logits, depths, images])
					print("loss " + str(loss))
					
					pred = outputDepth
					print("np.max(pred[0,:,:,0])):")
					print( np.max(pred[0,:,:,0]) )
					formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
					img = Image.fromarray(formatted)
					img.save("./output.png")

					pred = testDepth 
					print("np.max(pred[0,:,:,0])):")
					print( np.max(pred[0,:,:,0]) )
					formatted = ((pred[0,:,:,0]) * 255 / np.max(pred[0,:,:,0])).astype('uint8')
					img = Image.fromarray(formatted)
					img.save("./outputDepth.png")
					print('image[0]')
					print(image[0])
					print(image[0].shape)
					image2 = np.zeros((512,512,3), 'uint8')
					print(image.shape)
					img = Image.fromarray(image[0].astype('uint8'))
					img.save("./outputOrgImage.jpg")
			#saver = tf.train.Saver()
			#saver.save(sess, './ch/ch')
		except Exception as e:
			coord.request_stop(e)
			print(e)
			raise e
		print("done")


def main(argv=None):  # pylint: disable=unused-argument
	if not os.path.exists(CHECKPOINT_DIR):
		os.makedirs(CHECKPOINT_DIR)
	inputNetwork = theNetwork()
	runIt(inputNetwork)

main()
