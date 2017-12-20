import tensorflow as tf
import network
import os
from PIL import Image
import numpy as np

logsDir = './logs/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

#TODO: How did they deal with invalid depths?
#TODO: make sure the depths are right.../255 ??? 
def csv_inputs(csv_file_path, batch_size, imageSize, depthImageSize):

    filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    # input
    jpg = tf.read_file(filename)
    image = tf.image.decode_jpeg(jpg, channels=3)
    image = tf.cast(image, tf.float32)       
    # target
    depth_png = tf.read_file(depth_filename)
    depth = tf.image.decode_png(depth_png, channels=1)
    depth = tf.cast(depth, tf.float32)
    depth = tf.div(depth, [255.0])
    #depth = tf.cast(depth, tf.int64)

    # resize
    image = tf.image.resize_images(image, imageSize)
    depth = tf.image.resize_images(depth, depthImageSize)
    invalid_depth = tf.sign(depth)
    # generate batch
    images, depths, invalid_depths = tf.train.batch(
        [image, depth, invalid_depth],
        batch_size=batch_size,
        num_threads=4,
        capacity= 50 + 3 * batch_size,
    )
    return images, depths, invalid_depths


def loss(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / 55.0*74.0 - 0.5*sqare_sum_d / math.pow(55*74, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



def train(total_loss, global_step, batch_size):
    num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def runIt():
	if not os.path.exists(logsDir):
		os.makedirs(logsDir)

	IMAGE_HEIGHT = 228
	IMAGE_WIDTH = 304
	TARGET_HEIGHT = 55
	TARGET_WIDTH = 74
	BATCH_SIZE = 1#8
    images, depths, invalid_depths = csv_inputs(TRAIN_FILE, BATCH_SIZE, 
    	imageSize=(IMAGE_HEIGHT, IMAGE_WIDTH), depthImageSize=(TARGET_HEIGHT, TARGET_WIDTH))

	with tf.Session() as sess:
		logits = network.inference(images)
        loss = model.loss(logits, depths, invalid_depths)
		train_op

		file_writer = tf.summary.FileWriter(logsDir)
		file_writer.add_graph(sess.graph)


runIt()

