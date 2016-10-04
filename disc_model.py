import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq


import numpy as np

class DiscModel():
    def __init__(self, args):
        self.args = args

        if args.disc_model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.disc_model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.disc_model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))
        
        self.embedding = tf.Variable(tf.random_uniform([self.args.vocab_size, self.args.rnn_size], minval=-.05, maxval=.05, dtype=tf.float32), name='embedding')
        with tf.variable_scope('DISC') as scope:
            cell = cell_fn(args.rnn_size)

            self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
            # If the input data is given as word tokens, feed this value
            self.input_data_text = tf.placeholder(tf.int32, [args.batch_size, args.seq_length], name='input_data_text')
            #self.input_data_text = tf.Variable(tf.zeros((args.batch_size, args.seq_length), dtype=tf.int32), name='input_data_text')

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            # Fully connected layer is applied to the final state to determine the output class
            self.fc_layer = tf.Variable(tf.random_normal([args.rnn_size, 1], stddev=0.35, dtype=tf.float32), name='disc_fc_layer')
            self.lr = tf.Variable(0.0, trainable=False, name='learning_rate')
            self.has_init_seq2seq = False

    def attach_cost(self, gen_model):
        # TODO: Shouldn't dynamic RNN be used here?
        # output_text, states_text = rnn.rnn(cell, inputs, initial_state=self.initial_state)
        predicted_classes_text = self.discriminate_text(self.input_data_text)
        self.loss_text = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predicted_classes_text, np.ones((self.args.batch_size, 1), dtype=np.float32)))
        generated_wv = gen_model.generate()
        predicted_classes_wv = self.discriminate_wv(generated_wv)
        self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predicted_classes_wv, np.zeros((self.args.batch_size, 1), dtype=np.float32)))
        self.loss = .5 * self.loss_gen + .5 * self.loss_text
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
            self.args.grad_clip)
        # optimize only discriminator owned variables 
        g_and_v = [(g, v) for g, v in zip(grads, tvars) if v.name.startswith('DISC') or v.name.startswith('embedding')]
        print '##########'
        for _, v in g_and_v:
            print v.name
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(g_and_v)

    def discriminate_wv(self, input_data_wv):
        with tf.variable_scope('DISC', reuse=self.has_init_seq2seq) as scope:
            self.has_init_seq2seq = True
            output_wv, states_wv = seq2seq.rnn_decoder(input_data_wv, self.initial_state, self.cell, scope=scope)
            predicted_classes_wv = tf.matmul(output_wv[-1], self.fc_layer)
        return predicted_classes_wv

    def discriminate_text(self, input_data_text):
        inputs = tf.split(1, self.args.seq_length, tf.nn.embedding_lookup(self.embedding, input_data_text))
        inputs = map(lambda i: tf.nn.l2_normalize(i, 1), [tf.squeeze(input_, [1]) for input_ in inputs])
        return self.discriminate_wv(inputs)
