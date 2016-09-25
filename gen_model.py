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

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = tf.placeholder(tf.float32, [args.batch_size, args.rnn_size * 4], name="initial_state")
        
        self.lr = tf.Variable(0.0, trainable=False)
        self.has_init_seq2seq = False

    def attach_cost(self, disc_model):
        self.embedding = disc_model.embedding
        self.outputs = self.generate(self.initial_state, self.input_data)
        self.loss = 1. - tf.reduce_mean(disc_model.discriminate_wv(self.outputs))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                self.args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def generate(self, initial_state, input_data):
        for v in tf.get_collection(tf.GraphKeys.VARIABLES):
            print v.name
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
        return outputs

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        prime = list(vocab.keys())[2]
        print (prime)
        for word in [prime]:
            print (word)
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        word = prime
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = words[sample]
            ret += ' ' + pred
            word = pred
        return ret


