import tensorflow as tf
import os

with tf.Graph().as_default() as g:
  with tf.Session() as sess:
    x = tf.Variable(0., name="x", validate_shape=False)
    saver = tf.train.Saver() #now it is satisfied...
    saver.restore(sess, os.path.join(os.getcwd(), 'model'))
    x_ = sess.run(x)
    print(x_)