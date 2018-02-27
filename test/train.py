import tensorflow as tf
import numpy as np

def get_padded_batch(file_list, batch_size, input_size):
  file_queue = tf.train.string_input_producer(file_list)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[],
                                           dtype=tf.int64)}

  _, sequence = tf.parse_single_sequence_example(
      serialized_example, sequence_features=sequence_features)

  length = tf.shape(sequence['inputs'])[0]

  queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, input_size), (None,), ()])

  enqueue_ops = queue.enqueue([sequence['inputs'],
                                sequence['labels'],
                                length])
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue_ops]))
  return queue.dequeue_many(batch_size)

def build_graph(mode):
  batch_size = 4
  num_classes = 4
  input_size = num_classes
  num_cells = 2
  files = ['dataset']

  # Graph
  with tf.Graph().as_default() as graph:
    inputs, labels, lengths, = None, None, None
    state_is_tuple = True

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = get_padded_batch(files, batch_size, input_size)
    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [batch_size, None, input_size], name='inputs')
      # If state_is_tuple is True, the output RNN cell state will be a tuple
      # instead of a tensor. During training and evaluation this improves
      # performance. However, during generation, the RNN cell state is fed
      # back into the graph with a feed dict. Feed dicts require passed in
      # values to be tensors and not tuples, so state_is_tuple is set to False.
      state_is_tuple = False

    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_cells, state_is_tuple=state_is_tuple)
    initial_state = lstm.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        lstm, inputs, lengths, initial_state, parallel_iterations=1,
        swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, num_cells])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)
    tf.add_to_collection('logits', tf.reshape(logits_flat, (batch_size, -1, num_classes)))

    if mode == 'train' or mode == 'eval':
      labels_flat = tf.reshape(labels, [-1])
      softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_flat, logits=logits_flat)
      loss = tf.reduce_mean(softmax_cross_entropy)
      perplexity = tf.reduce_mean(tf.exp(softmax_cross_entropy))

      correct_predictions = tf.to_float(
          tf.nn.in_top_k(logits_flat, labels_flat, 1))
      accuracy = tf.reduce_mean(correct_predictions) * 100

      global_step = tf.Variable(0, trainable=False, name='global_step')

      tf.add_to_collection('loss', loss)
      tf.add_to_collection('perplexity', perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      summaries = [
          tf.summary.scalar('loss', loss),
          tf.summary.scalar('perplexity', perplexity),
          tf.summary.scalar('accuracy', accuracy)
      ]

      dummy = tf.Variable(3.)

      if mode == 'train':
        learning_rate = tf.train.exponential_decay(
            0.01, global_step, 100,
            0.99, staircase=True, name='learning_rate')

        opt = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step)
        tf.add_to_collection('learning_rate', learning_rate)
        tf.add_to_collection('train_op', train_op)

        summaries.append(tf.summary.scalar(
            'learning_rate', learning_rate))

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [], name='temperature')
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)

  return graph


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  graph = build_graph('train')
  summary_frequency = 1
  num_training_steps = 600
  files = ['dataset']
  batch_size = 4
  input_size = 4

  with graph.as_default():
    with tf.Session() as sess:
      global_step = graph.get_collection('global_step')[0]
      learning_rate = graph.get_collection('learning_rate')[0]
      loss = graph.get_collection('loss')[0]
      perplexity = graph.get_collection('perplexity')[0]
      accuracy = graph.get_collection('accuracy')[0]
      train_op = graph.get_collection('train_op')[0]
      logits = graph.get_collection('logits')[0]

      coord = tf.train.Coordinator()
      saver = tf.train.Saver()

      sess.run(tf.global_variables_initializer())
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      global_step_ = sess.run(global_step)

      tf.logging.info('Starting training loop...')

      while not num_training_steps or global_step_ < num_training_steps:
        if global_step_ + 1 == num_training_steps:
          (global_step_, learning_rate_, loss_, perplexity_, accuracy_, logits_,
           _) = sess.run([global_step, learning_rate, loss, perplexity, accuracy, logits,
                          train_op])
          tf.logging.info('Global Step: %d - '
                          'Learning Rate: %.5f - '
                          'Loss: %.3f - '
                          'Perplexity: %.3f - '
                          'Accuracy: %.3f',
                          global_step_, learning_rate_, loss_, perplexity_,
                          accuracy_)
          tf.logging.info(logits_)
        if (global_step_) % 100 == 0:
          (global_step_, learning_rate_, loss_, perplexity_, accuracy_,
           _) = sess.run([global_step, learning_rate, loss, perplexity, accuracy,
                          train_op])
          tf.logging.info('Global Step: %d - '
                          'Learning Rate: %.5f - '
                          'Loss: %.3f - '
                          'Perplexity: %.3f - '
                          'Accuracy: %.3f',
                          global_step_, learning_rate_, loss_, perplexity_,
                          accuracy_)
        else:
          (global_step_, _) = sess.run([global_step, train_op])

      # Stop queue runners
      coord.request_stop()
      coord.join(threads)

      checkpoint_file = saver.save(sess, "model.ckpt", write_meta_graph=False)
      tf.logging.info('Training complete. Saving: %s', checkpoint_file)

  graph = build_graph('generate')
  with graph.as_default():
    with tf.Session() as sess:
      # Restore variables
      saver = tf.train.Saver()
      checkpoint_file = tf.train.latest_checkpoint(".")
      tf.logging.info("Restoring: %s", checkpoint_file)
      saver.restore(sess, checkpoint_file)

      # Get variables from graph
      inputs = graph.get_collection('inputs')[0]
      initial_state = graph.get_collection('initial_state')[0]
      final_state = graph.get_collection('final_state')[0]
      temperature = graph.get_collection('temperature')[0]
      softmax = graph.get_collection('softmax')[0]
      logits = graph.get_collection('logits')[0]

      # Initialize stuff
      coord = tf.train.Coordinator()
      sess.run(tf.global_variables_initializer())

      # Get the data from SequenceExample
      data, labels, lengths = get_padded_batch(files, batch_size, input_size)
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      inputs_ = sess.run(tf.expand_dims(data[:, 1], 1))
      print(inputs_)

      # Get initial state from graph
      state_ = sess.run(initial_state)

      results = np.argmax(inputs_, axis=2)
      all_logits = np.zeros((batch_size, 1, input_size))

      # Pass through RNN
      for i in xrange(9):
        (softmax_, logits_, state_) = sess.run((softmax, logits, final_state), feed_dict={ 
          inputs: inputs_, 
          initial_state: state_,
          temperature: 1.0 })

        next_result = np.argmax(softmax_, axis=2)
        results = np.append(results, next_result, axis=1)
        all_logits = np.append(all_logits, logits_, axis=1)

        inputs_ = np.zeros((batch_size, 1, input_size))
        inputs_[np.arange(batch_size), 0, next_result[:, 0]] = 1

      print(all_logits)


      print(results)

      # Stop queue runners
      coord.request_stop()
      coord.join(threads)

        

if __name__ == '__main__':
  tf.app.run(main)