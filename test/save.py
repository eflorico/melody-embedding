import tensorflow as tf

with tf.Graph().as_default() as g:#yes you have to have a graph first
  with tf.Session() as sess:
    x = tf.Variable(tf.zeros([3,2]), name="x")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print(saver.save(sess,'model'))