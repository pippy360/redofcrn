import numpy as np
import tensorflow as tf


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 30
INITIAL_LEARNING_RATE = 0.000000001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999

def getFilenameQueuesFromCSVFile( csvFilePath ):
    filename_queue = tf.train.string_input_producer([csvFilePath], shuffle=False)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    return filename, depth_filename

def csv_inputs(image_filename, depth_filename, batch_size, imageSize, depthImageSize):

    # input
    image = tf.read_file(image_filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, imageSize)
    
    # target
    depth = tf.read_file(depth_filename)
    depth = tf.image.decode_png(depth, channels=3)#fixme: change to 1 channel
    print('depth.shape')
    print(depth.shape)
    depth = tf.image.resize_images(depth, depthImageSize)
    print(depth.shape)
    depth = tf.slice(depth, [0,0, 0], [-1,-1,1])
    depth = tf.divide(depth, 100.0)

    # resize
    invalid_depth = tf.sign(depth)
    # generate batch
    images, depths, invalid_depths, filenames = tf.train.batch(
        [image, depth, invalid_depth, image_filename],
        batch_size=batch_size,
        num_threads=4,
        capacity= 50 + 3 * batch_size,
    )
    tf.summary.image('input_images', images, max_outputs=3)
    return images, depths, invalid_depths, filenames


#FIXME: REMOVE THIS 
IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 128
TARGET_WIDTH = 160
BATCH_SIZE = 8

def loss_scale_invariant_l2_norm(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, TARGET_HEIGHT*TARGET_WIDTH])
    depths_flat = tf.reshape(depths, [-1, TARGET_HEIGHT*TARGET_WIDTH])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, TARGET_HEIGHT*TARGET_WIDTH])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / TARGET_HEIGHT*TARGET_WIDTH - 0.5*sqare_sum_d / math.pow(TARGET_HEIGHT*TARGET_WIDTH, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_l2_norm(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, TARGET_HEIGHT*TARGET_WIDTH])
    depths_flat = tf.reshape(depths, [-1, TARGET_HEIGHT*TARGET_WIDTH])

    d = tf.subtract(logits_flat, depths_flat)
    return tf.nn.l2_loss(d)

def residualLayer(name, input_data, inputSize, outputSize, isResize, strides=None):
	with tf.variable_scope(name) as scope:
            if strides == None:
                if isResize:
                    strides = (2, 2)
                else:
                    strides = (1, 1)

            #conv 1x1 /1
            conv = input_data
            conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=1, strides=strides, padding='SAME', use_bias=False)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.nn.relu(conv)

            #conv 3x3 /1
            conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=3, padding='SAME', use_bias=False)
            conv = tf.layers.batch_normalization(conv)
            conv = tf.nn.relu(conv)

            #conv 1x1 /1
            conv = tf.layers.conv2d(conv, filters=outputSize, kernel_size=1, padding='SAME', use_bias=False)
            conv = tf.layers.batch_normalization(conv)
            #NO RELU
            
            #residule stuff
            inputsToAdd = input_data
            if(isResize):		
                    inputsToAdd = tf.layers.conv2d(inputsToAdd, filters=outputSize, kernel_size=1, strides=strides, padding='SAME', use_bias=False)
                    inputsToAdd = tf.layers.batch_normalization(inputsToAdd)

            conv = tf.add(conv, inputsToAdd)
            conv = tf.nn.relu(conv)
            return conv


def inference(input_images, keep_prob=.5):
	#input 304 x 228 x 3

	conv1 = tf.layers.conv2d(input_images, filters=64, kernel_size=7, strides=(2, 2), padding='SAME', name='conv1', use_bias=False)
	#Using biases
	biases = tf.get_variable('biases1', [64], dtype='float32', trainable=True)
	conv1 = tf.nn.bias_add(conv1, biases)
        
	conv2 = tf.layers.batch_normalization(conv1, name='bn_conv1')
	conv3 = tf.nn.relu(conv2)
	conv4 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=(2, 2), name='pool1', padding='SAME')
	
	conv5 = residualLayer("Res_resize_1_", conv4, inputSize=64, outputSize=256, isResize=True, strides=(1, 1))
	for i in range(2):
		conv5 = residualLayer("Res_1_"+str(i), conv5, inputSize=64, outputSize=256, isResize=False)

	conv6 = residualLayer("Res_resize_2_", conv5, inputSize=128, outputSize=512, isResize=True)
	for i in range(3):
		conv6 = residualLayer("Res_2_"+str(i), conv6, inputSize=128, outputSize=512, isResize=False)

	conv7 = residualLayer("Res_resize_3_", conv6, inputSize=256, outputSize=1024, isResize=True)
	for i in range(5):
		conv7 = residualLayer("Res_3_"+str(i), conv7, inputSize=256, outputSize=1024, isResize=False)

	conv8 = residualLayer("Res_resize_4_", conv7, inputSize=512, outputSize=2048, isResize=True)
	for i in range(2):
		conv8 = residualLayer("Res_4_"+str(i), conv8, inputSize=512, outputSize=2048, isResize=False)

	conv9 = tf.layers.conv2d(conv8, filters=1024, kernel_size=1, padding='SAME', use_bias=False)
	#Using biases
	biases = tf.get_variable('biases2', [1024], dtype='float32', trainable=True)
	conv9 = tf.nn.bias_add(conv9, biases)

	conv10 = tf.layers.batch_normalization(conv9)

	conv11 = up_project(conv10, kernel_size=3, filters_size=512, id="2x")
	conv12 = up_project(conv11, kernel_size=3, filters_size=256, id="4x")
	conv13 = up_project(conv12, kernel_size=3, filters_size=128, id="8x")
	conv14 = up_project(conv13, kernel_size=3, filters_size=64, id="16x")

	conv15 = tf.nn.dropout(conv14, keep_prob=keep_prob)#Change to 1. while testing and .5 while training
	conv16 = tf.layers.conv2d(conv15, filters=1, kernel_size=3, padding='SAME')
	#Note: In the original model a single bias value is used here for the convolution which seems pointless

	return conv16


####
#		upconvolution
####


def get_incoming_shape(incoming):
	""" Returns the incoming data shape """
	if isinstance(incoming, tf.Tensor):
		return incoming.get_shape().as_list()
	elif type(incoming) in [np.array, list, tuple]:
		return np.shape(incoming)
	else:
		raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
	old_shape = get_incoming_shape(tensors[0])[1:]
	new_shape = [-1] + old_shape
	new_shape[axis] *= len(tensors)
	return tf.reshape(tf.stack(tensors, axis + 1), new_shape)


def unpool_as_conv(input_data, convOutputSize, id, ReLU=False, BN=True):

	# Model upconvolutions (unpooling + convolution) as interleaving feature
	# maps of four convolutions (A,B,C,D). Building block for up-projections. 


	# Convolution A (3x3)
	# --------------------------------------------------
	layerName = "layer%s_ConvA" % (id)
	outputA = tf.layers.conv2d(input_data, filters=convOutputSize, kernel_size=(3, 3), padding='SAME', name=layerName, use_bias=False)
	#Using biases
	biases = tf.get_variable("layer%s_ConvA_biases" % (id), [convOutputSize], dtype='float32', trainable=True)
	outputA = tf.nn.bias_add(outputA, biases)

	# Convolution B (2x3)
	# --------------------------------------------------
	layerName = "layer%s_ConvB" % (id)
	padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
	outputB = tf.layers.conv2d(padded_input_B, filters=convOutputSize, kernel_size=(2, 3), padding='VALID', name=layerName, use_bias=False)
	#Using biases
	biases = tf.get_variable("layer%s_ConvB_biases" % (id), [convOutputSize], dtype='float32', trainable=True)
	outputB = tf.nn.bias_add(outputB, biases)

	# Convolution C (3x2)
	# --------------------------------------------------
	layerName = "layer%s_ConvC" % (id)
	padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
	outputC = tf.layers.conv2d(padded_input_C, filters=convOutputSize, kernel_size=(3, 2), padding='VALID', name=layerName, use_bias=False)
	#Using biases
	biases = tf.get_variable("layer%s_ConvC_biases" % (id), [convOutputSize], dtype='float32', trainable=True)
	outputC = tf.nn.bias_add(outputC, biases)

	# Convolution D (2x2)
	# --------------------------------------------------
	layerName = "layer%s_ConvD" % (id)
	padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
	outputD = tf.layers.conv2d(padded_input_D, filters=convOutputSize, kernel_size=(2, 2), padding='VALID', name=layerName, use_bias=False)
	#Using biases
	biases = tf.get_variable("layer%s_ConvD_biases" % (id), [convOutputSize], dtype='float32', trainable=True)
	outputD = tf.nn.bias_add(outputD, biases)

	# Interleaving elements of the four feature maps
	# --------------------------------------------------
	left = interleave([outputA, outputB], axis=1)  # columns
	right = interleave([outputC, outputD], axis=1)  # columns
	result = interleave([left, right], axis=2) # rows

	if BN:
		layerName = "layer%s_BN" % (id)
		result = tf.layers.batch_normalization(result, name=layerName)

	if ReLU:#TODO: TAKE THIS OUTSIDE THIS FUNCTION
		result = tf.nn.relu(result, name=layerName)

	return result


def up_project(input_data, kernel_size, filters_size, id):

	# Create residual upsampling layer (UpProjection)

	# Branch 1
	# Interleaving Convs of 1st branch
	id_br1 = "%s_br1" % (id)
	branch1_out1 = unpool_as_conv(input_data, convOutputSize=filters_size, id=id_br1, ReLU=True, BN=True)

	# Convolution following the upProjection on the 1st branch
	branch1_out2 = tf.layers.conv2d(branch1_out1, filters=filters_size, kernel_size=kernel_size, padding='SAME', use_bias=False)
	#Using biases
	biases = tf.get_variable('biases_upproject_'+str(id)+'_1', [filters_size], dtype='float32', trainable=True)
	branch1_out2 = tf.nn.bias_add(branch1_out2, biases)

	layerName = "layer%s_BN" % (id)
	branch1_out3 = tf.layers.batch_normalization(branch1_out2, name=layerName)

	# Output of 1st branch
	branch1_output = branch1_out3

	# Branch 2
	# Interleaving convolutions and output of 2nd branch
	id_br2 = "%s_br2" % (id)
	branch2_output = unpool_as_conv(input_data, convOutputSize=filters_size, id=id_br2, ReLU=False, BN=True)

	# sum branches
	layerName = "layer%s_Sum" % (id)
	output = tf.add_n([branch1_output, branch2_output], name=layerName)
	# ReLU
	layerName = "layer%s_ReLU" % (id)
	output = tf.nn.relu(output, name=layerName)

	return output


class Network:

	def getInference(self, images):
		return inference(images)

	def getCheckpointDir(self):
		return './ch'

	def restore(self, sess, restoreVars):

		#TODO: handle case when no checkpoint

		# Restore the moving average version of the learned variables for eval.
		variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()

		#is there one??
		ckpt = tf.train.get_checkpoint_state( self.getCheckpointDir() )
		if ckpt and ckpt.model_checkpoint_path:
			saver = tf.train.Saver()
			# Restores from checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			# Assuming model_checkpoint_path looks something like:
			#   /my-favorite-path/cifar10_train/model.ckpt-0,
			# extract global_step from it.
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

			print('checkpoint loaded with global_step: ' + str(global_step))
			return global_step
		else:
			return None
