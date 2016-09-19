import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

import numpy as np

class DiscModel():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.disc_model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.disc_model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.disc_model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
        # If the input data is given as word tokens, feed this value
        self.input_data_text = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        # If the input data is given as the generator's word vec outputs, feed this value
        self.input_data_wv = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.rnn_size])
        # Binary targets. 1 = is valid, 0 = came from generator
        self.targets = tf.placeholder(tf.float32, [args.batch_size, 1])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        # Fully connected layer is applied to the final state to determine the output class
        self.final_weights = tf.Variable(tf.random_normal([args.rnn_size, 1], stddev=0.35),
            name="final_weights")
        with tf.variable_scope('rnnlm'):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data_text))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # TODO: Shouldn't dynamic RNN be used here?
        #output_text, states_text = rnn.rnn(cell, inputs, initial_state=self.initial_state)
        inputs_wv = tf.split(1, args.seq_length, self.input_data_wv)
        inputs_wv = [tf.squeeze(input_, [1]) for input_ in inputs_wv]
        output_wv, states_wv = rnn.rnn(cell, inputs_wv, initial_state=self.initial_state)

        #predicted_classes_text = tf.matmul(output_text[-1], self.final_weights)
        predicted_classes_wv = tf.matmul(output_wv[-1], self.final_weights)

        #loss_text = tf.nn.sigmoid_cross_entropy_with_logits(predicted_classes_text, self.targets)
        loss_wv = tf.nn.sigmoid_cross_entropy_with_logits(predicted_classes_wv, self.targets)

        #self.cost_text = tf.reduce_mean(loss_text)
        self.cost_wv = tf.reduce_mean(loss_wv)
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_wv, tvars),
                args.grad_clip)
        optimizer_wv = tf.train.AdamOptimizer(self.lr)
        self.train_op_wv = optimizer_wv.apply_gradients(zip(grads, tvars))
        
        '''
        grads_text, _ = tf.clip_by_global_norm(tf.gradients(self.cost_text, tvars),
                args.grad_clip)
        optimizer_text = tf.train.AdamOptimizer(self.lr)
        self.train_op_text = optimizer_text.apply_gradients(zip(grads_text, tvars))
        '''
