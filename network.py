import tensorflow as tf

def purpleOne(name, inputs, inputSize, outputSize, isPurple):
	with tf.variable_scope(name) as scope:
	    #conv 1x1 /1
		conv = inputs
		conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=1, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		conv = tf.nn.relu(conv)

		#conv 3x3 /1
		conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=3, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		conv = tf.nn.relu(conv)

		#conv 1x1 /1
		conv = tf.layers.conv2d(conv, outputSize, 1, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		#NO RELU
		
		#residule stuff
		inputsToAdd = inputs
		if(isPurple)		
			inputsToAdd = tf.layers.conv2d(inputsToAdd, outputSize, 1, padding='SAME')
			inputsToAdd = tf.layers.batch_normalization(inputsToAdd)

		conv = tf.add(conv, inputsToAdd)
		conv = tf.nn.relu(conv)
		return conv


def inference(input_data):
	#input 304 x 228 x 3

	conv1 = tf.layers.conv2d(input_data, filters=64, kernel_size=7, padding='SAME')
	conv2 = tf.layers.batch_normalization(conv1)
	conv3 = tf.nn.relu(conv2)
	conv4 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=2)

	conv5 = purpleOne("purple1_", conv4, inputSize=64, outputSize=256)
	for i in range(2):
		conv5 = purpleOne("blue1_"+str(i), conv5, inputSize=64, outputSize=256, isPurple=False)

	conv6 = purpleOne("purple2_", conv5, inputSize=128, outputSize=512)
	for i in range(3):
		conv6 = purpleOne("blue2_"+str(i), conv6, inputSize=128, outputSize=512, isPurple=False)

	conv7 = purpleOne("purple3_", conv6, inputSize=256, outputSize=1024)
	for i in range(5):
		conv7 = purpleOne("blue3_"+str(i), conv7, inputSize=256, outputSize=1024, isPurple=False)

	conv8 = purpleOne("purple4_", conv7, inputSize=512, outputSize=2048)
	for i in range(2):
		conv8 = purpleOne("blue4_"+str(i), conv8, inputSize=512, outputSize=2048, isPurple=False)

	conv9 = tf.layers.conv2d(conv8, filters=1024, kernel_size=1, padding='SAME')
	conv10 = tf.layers.batch_normalization(conv9, name = layerName)

	conv11 = up_project(conv10, kernel_size=3, filters_size=512, id=0)
	conv12 = up_project(conv11, kernel_size=3, filters_size=256, id=0)
	conv13 = up_project(conv12, kernel_size=3, filters_size=128, id=0)
	conv14 = up_project(conv13, kernel_size=3, filters_size=64, id=0)

	conv15 = tf.layers.conv2d(conv14, filters=1, kernel_size=3, padding='SAME')

	return conv15


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


def unpool_as_conv(input_data, convOutputSize, id, ReLU = False, BN = True):

	# Model upconvolutions (unpooling + convolution) as interleaving feature
	# maps of four convolutions (A,B,C,D). Building block for up-projections. 


	# Convolution A (3x3)
	# --------------------------------------------------
	layerName = "layer%s_ConvA" % (id)
    outputA = tf.layers.conv2d(input_data, filters=convOutputSize, kernel_size=(3, 3), padding='SAME', name=layerName)

	# Convolution B (2x3)
	# --------------------------------------------------
	layerName = "layer%s_ConvB" % (id)
	padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
    outputB = tf.layers.conv2d(padded_input_B, filters=convOutputSize, kernel_size=(2, 3), padding='VALID', name=layerName)

	# Convolution C (3x2)
	# --------------------------------------------------
	layerName = "layer%s_ConvC" % (id)
	padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
	outputC = tf.layers.conv2d(padded_input_C, filters=convOutputSize, kernel_size=(3, 2), padding='VALID', name=layerName)

	# Convolution D (2x2)
	# --------------------------------------------------
	layerName = "layer%s_ConvD" % (id)
	padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
	outputD = tf.layers.conv2d(padded_input_D, filters=convOutputSize, kernel_size=(2, 2), padding='VALID', name=layerName)

	# Interleaving elements of the four feature maps
	# --------------------------------------------------
	left = interleave([outputA, outputB], axis=1)  # columns
	right = interleave([outputC, outputD], axis=1)  # columns
	result = interleave([left, right], axis=2) # rows

	if BN:
		layerName = "layer%s_BN" % (id)
		result = tf.layers.batch_normalization(result, name=layerName)

	if ReLU:#TODO: TAKE THIS OUTSIDE THIS FUNCTION
		result = tf.nn.relu(result, name = layerName)

	return result


def up_project(input_data, kernel_size, filters_size, id):

	# Create residual upsampling layer (UpProjection)

	# Branch 1
	# Interleaving Convs of 1st branch
	id_br1 = "%s_br1" % (id)
	branch1_out1 = unpool_as_conv(input_data, convOutputSize=filters_size, ReLU=True, BN=True)

	# Convolution following the upProjection on the 1st branch
	branch1_out2 = tf.layers.conv2d(branch1_out1, filters=filters_size, kernel_size=kernel_size, padding='SAME')

	layerName = "layer%s_BN" % (id)
	branch1_out3 = tf.layers.batch_normalization(branch1_out2, name = layerName)

	# Output of 1st branch
	branch1_output = branch1_out3

	# Branch 2
	# Interleaving convolutions and output of 2nd branch
	id_br2 = "%s_br2" % (id)
	branch2_output = unpool_as_conv(input_data, convOutputSize=filters_size, ReLU=False, BN=True)

	# sum branches
	layerName = "layer%s_Sum" % (id)
	output = tf.add_n([branch1_output, branch2_output], name = layerName)
	# ReLU
	layerName = "layer%s_ReLU" % (id)
	output = tf.nn.relu(output, name=layerName)

	return out
