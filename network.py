import tensorflow as tf

def purpleOne(name, inputs, inputSize, outputSize, isPurple):
	with tf.variable_scope(name) as scope:
	    #conv 1x1 /1
		conv = inputs
		conv = tf.layers.conv2d(conv, filters=inputSize, kernel_size=1, padding='SAME', use_bias)
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


def inference(inputs):
	#input 304 x 228 x 3

	conv = tf.layers.conv2d(inputs, filters=64, kernel_size=7, padding='SAME')
	conv = tf.layers.batch_normalization(conv)
	conv = tf.nn.relu(conv)
	conv = tf.layers.max_pooling2d(conv, pool_size=3, strides=2)

	conv = purpleOne("purple1_", conv, inputSize=64, outputSize=256)
	for i in range(2):
		conv = purpleOne("blue1_"+str(i), conv, inputSize=64, outputSize=256, isPurple=False)

	conv = purpleOne("purple2_", conv, inputSize=128, outputSize=512)
	for i in range(3):
		conv = purpleOne("blue2_"+str(i), conv, inputSize=128, outputSize=512, isPurple=False)

	conv = purpleOne("purple3_", conv, inputSize=256, outputSize=1024)
	for i in range(5):
		conv = purpleOne("blue3_"+str(i), conv, inputSize=256, outputSize=1024, isPurple=False)

	conv = purpleOne("purple4_", conv, inputSize=512, outputSize=2048)
	for i in range(2):
		conv = purpleOne("blue4_"+str(i), conv, inputSize=512, outputSize=2048, isPurple=False)

	return conv


def unpool_as_conv(input_data, size, stride = 1):

	# Model upconvolutions (unpooling + convolution) as interleaving feature
	# maps of four convolutions (A,B,C,D). Building block for up-projections. 


	# Convolution A (3x3)
	# --------------------------------------------------
	layerName = "layer%s_ConvA" % (id)
	self.feed(input_data)
	self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False)
	outputA = self.get_output()

	# Convolution B (2x3)
	# --------------------------------------------------
	layerName = "layer%s_ConvB" % (id)
	padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
	self.feed(padded_input_B)
	self.conv(2, 3, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
	outputB = self.get_output()

	# Convolution C (3x2)
	# --------------------------------------------------
	layerName = "layer%s_ConvC" % (id)
	padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
	self.feed(padded_input_C)
	self.conv(3, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
	outputC = self.get_output()

	# Convolution D (2x2)
	# --------------------------------------------------
	layerName = "layer%s_ConvD" % (id)
	padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
	self.feed(padded_input_D)
	self.conv(2, 2, size[3], stride, stride, name = layerName, padding = 'VALID', relu = False)
	outputD = self.get_output()

	# Interleaving elements of the four feature maps
	# --------------------------------------------------
	left = interleave([outputA, outputB], axis=1)  # columns
	right = interleave([outputC, outputD], axis=1)  # columns
	Y = interleave([left, right], axis=2) # rows

	return Y


def up_project(inputs, size, id, stride = 1):

	# Create residual upsampling layer (UpProjection)

	inputs = self.get_output()

	# Branch 1
	# Interleaving Convs of 1st branch
	out = unpool_as_conv(inputs, size, stride)
	ReLU=True, 
	BN=True


	# Convolution following the upProjection on the 1st branch
	self.conv(size[0], size[1], size[3], stride, stride, relu = False)

	layerName = "layer%s_BN" % (id)
	self.batch_normalization(name = layerName, scale_offset=True, relu = False)

	# Output of 1st branch
	branch1_output = self.get_output()

	# Branch 2
	# Interleaving convolutions and output of 2nd branch
	branch2_output = self.unpool_as_conv(inputs, size, stride, ReLU=False)
	BN=True

	# sum branches
	layerName = "layer%s_Sum" % (id)
	output = tf.add_n([branch1_output, branch2_output], name = layerName)
	# ReLU
	layerName = "layer%s_ReLU" % (id)
	output = tf.nn.relu(output, name=layerName)

	self.feed(output)
	return self
