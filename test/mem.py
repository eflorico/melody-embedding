import tensorflow as tf
import numpy as np

# Toy problem: minimize x^2, for four different x's

all_x = np.array([-3.,-1,1,3])
all_m = np.zeros(4)
all_v = np.zeros(4)

with tf.Graph().as_default() as g:
	with tf.Session() as sess:
		# For some reason, training files if I try to initialize x directly
		x_in = tf.placeholder(tf.float32, shape=())
		x = tf.Variable(x_in)

		params = [x]
		loss = x*x
		gradients = tf.gradients(loss, params)

		opt = tf.train.AdamOptimizer(0.1)
		train_op = opt.apply_gradients(zip(gradients, params))

		# Adam internal states
		m = tf.placeholder(tf.float32)
		v = tf.placeholder(tf.float32)
		assign_m = opt.get_slot(x, 'm').assign(m)
		assign_v = opt.get_slot(x, 'v').assign(v)

		# Initialize with some random value for the placeholder so that TensorFlow doesn't complain
		sess.run(tf.global_variables_initializer(), {x_in:-999})

		# Training steps
		for i in xrange(100):
			# Alternatingly train 4 different x's
			for j in xrange(4):
				# Load Adam state
				sess.run([assign_m, assign_v], {m: all_m[j], v: all_v[j]})

				# Load x
				sess.run(tf.variables_initializer([x]), {x_in: all_x[j]})

				# Run Adam
				sess.run([train_op])

				# Get state back out
				all_x[j], all_m[j], all_v[j] = sess.run([x, opt.get_slot(x, 'm'), opt.get_slot(x, 'v')])
		
		print all_x