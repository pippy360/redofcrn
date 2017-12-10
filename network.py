import tensorflow as tf

def purpleOne(name, inputs, inputSize, outputSize):
	with tf.variable_scope(name) as scope:
	    #conv 1x1 /1
		conv = inputs
		conv = tf.layers.conv2d(conv, inputSize, 1, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		conv = tf.nn.relu(conv)
		#TODO: relu

		#conv 3x3 /1
		conv = tf.layers.conv2d(conv, inputSize, 3, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		conv = tf.nn.relu(conv)

		#conv 1x1 /1
		conv = tf.layers.conv2d(conv, outputSize, 1, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		#NO RELU
		
		#residule stuff
		inputsResized = tf.layers.conv2d(inputs, outputSize, 1, padding='SAME')
		inputsResized = tf.layers.batch_normalization(inputsResized)

		conv = conv + inputsResized
		conv = tf.nn.relu(conv)
		return conv

def blueOne(name, inputs, inputSize, outputSize):
	with tf.variable_scope(name) as scope:
		#conv 1x1 /1
		conv = inputs
		conv = tf.layers.conv2d(conv, inputSize, 1, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		conv = tf.nn.relu(conv)
		#TODO: relu

		#conv 3x3 /1
		conv = tf.layers.conv2d(conv, inputSize, 3, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		conv = tf.nn.relu(conv)

		#conv 1x1 /1
		conv = tf.layers.conv2d(conv, outputSize, 1, padding='SAME')
		conv = tf.layers.batch_normalization(conv)
		#NO RELU
		
		#residule stuff
		conv = conv + inputs
		return conv



def inference(inputs):
	#input 304 x 228 x 3

	conv = tf.layers.conv2d(inputs, 3, 7, padding='SAME')
	conv = tf.layers.batch_normalization(conv)
	conv = tf.layers.max_pooling2d(conv, pool_size=3, strides=2)
	conv = tf.nn.relu(conv)

	conv = purpleOne("purple1_", conv, 64, 256)
	for i in range(2):
		conv = blueOne("blue1_"+str(i), conv, 64, 256)

	conv = purpleOne("purple2_", conv, 128, 512)
	for i in range(3):
		conv = blueOne("blue2_"+str(i), conv, 128, 512)

	conv = purpleOne("purple3_", conv, 256, 1024)
	for i in range(5):
		conv = blueOne("blue3_"+str(i), conv, 256, 1024)

	conv = purpleOne("purple4_", conv, 512, 2048)
	for i in range(2):
		conv = blueOne("blue4_"+str(i), conv, 512, 2048)

	return conv



