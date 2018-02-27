import tensorflow as tf

# Training data
num_classes = 4
labels = [ [ i ] * 10 for i in xrange(num_classes) ]

writer = tf.python_io.TFRecordWriter('dataset')

with tf.Session() as sess:
  inputs = tf.one_hot(tf.constant(labels), num_classes).eval().tolist()
  print(inputs)

  for i in xrange(num_classes):
    # Sequence example
    input_features = [
        tf.train.Feature(float_list=tf.train.FloatList(value=input_))
        for input_ in inputs[i]]
    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        for label in labels[i]]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }

    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    ex = tf.train.SequenceExample(feature_lists=feature_lists)

    writer.write(ex.SerializeToString())

writer.close()