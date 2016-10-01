import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

# The purpose of this is to regularize the embedding.

class StandardModel():
    def __init__(self, args, embedding):
        self.args = args

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name='STAND_input')
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name='STAND_targets')
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.embedding = embedding
        with tf.variable_scope('STAND'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, i):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        o, _ = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='STAND')
        with tf.variable_scope('STAND', reuse=True) as scope:
            sf_o, _ = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop, scope=scope)
        output = tf.reshape(tf.concat(1, o), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        sf_output = tf.reshape(tf.concat(1, sf_o), [-1, args.rnn_size])
        self_feed_logits = tf.matmul(sf_output, softmax_w) + softmax_b
        self.self_feed_probs = tf.nn.softmax(self_feed_logits)

        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.loss = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                args.grad_clip)
        for g, v in zip(grads, tvars):
            print v.name
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
