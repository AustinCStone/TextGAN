import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class GenModel():
    def __init__(self, args):
        self.args = args

        if args.gen_model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.gen_model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.gen_model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        with tf.variable_scope('GEN') as scope:
            cell = cell_fn(args.rnn_size)

            self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.initial_state = tf.placeholder(tf.float32, [args.batch_size, args.rnn_size * 4], name="initial_state")
            self.outputs = tf.placeholder(tf.float32, [args.seq_length, args.batch_size, args.rnn_size])
            self.lr = tf.Variable(0.0, trainable=False, name='learning_rate')
            self.has_init_seq2seq = False

    def attach_cost(self, disc_model):
        self.embedding = disc_model.embedding
        self.outputs = self.generate(self.initial_state, self.input_data)
        self.loss = 1. - tf.reduce_mean(disc_model.discriminate_wv(self.outputs))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                self.args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        v_and_g = [(g, v) for g, v in zip(grads, tvars) if v.name.startswith('GEN')]
        self.train_op = optimizer.apply_gradients(v_and_g)

    def generate(self, initial_state, input_data):
        with tf.device("/cpu:0"):
            inputs = tf.split(1, self.args.seq_length, tf.nn.embedding_lookup(self.embedding, input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # TODO: Should we have some transition weights here?
        def loop(prev, _):
            return prev

        #print initial_state.get_shape()
        with tf.variable_scope('GEN', reuse=self.has_init_seq2seq) as scope:
            self.has_init_seq2seq = True
            outputs, last_state = seq2seq.rnn_decoder(inputs, initial_state, self.cell, loop_function=loop, scope=scope)
        self.outputs = outputs
        return outputs
        

