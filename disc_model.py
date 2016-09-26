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
        gen_model_initial_state = np.random.uniform(-1., 1., (self.args.batch_size, self.args.rnn_size)).astype('float32')
        gen_model_input_data = self.input_data_text
        generated_wv = gen_model.generate(gen_model_initial_state, gen_model_input_data)
        predicted_classes_wv = self.discriminate_wv(generated_wv)
        self.loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predicted_classes_wv, np.zeros((self.args.batch_size, 1), dtype=np.float32)))
        self.loss = .5 * self.loss_gen + .5 * self.loss_text
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
            self.args.grad_clip)
        # optimize only discriminator owned variables 
        g_and_v = [(g, v) for g, v in zip(grads, tvars) if v.name.startswith('DISC')]
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
        # TODO: Trying this out right now as a hack.. How to make this more principled? Adding noise to the disciminator word vectors
        # so it is harder for it to knock the generator for using the wrong word vecs
        #inputs = [tf.squeeze(input_, [1]) + np.random.normal(0., .01, (self.args.batch_size, self.args.rnn_size)) for input_ in inputs]
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        return self.discriminate_wv(inputs)
