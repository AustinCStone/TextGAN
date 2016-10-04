import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader, print_wv_nn, print_softmax
from gen_model import GenModel
from disc_model import DiscModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_size', type=int, default=10,
                        help='latent space size of the generator') 
    parser.add_argument('--data_dir', type=str, default='data/sherlock',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                        help='rnn, gru, or lstm')
    parser.add_argument('--gen_model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--disc_model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=30,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--vocab_size', type=int, default=20000,
                       help='max vocabulary size')                        
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to gen_model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--disc_train_bound', type=float, default=.3,
                       help='train the discriminator (only) until its loss reaches this bound')
    parser.add_argument('--gen_train_bound', type=float, default=1.0,
                       help='train the generator (only) until its loss reaches this bound')
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length, args.vocab_size)
    args.vocab_size = data_loader.vocab_size + 3 # plus 3 for unknown, end, and pad tokens
    
    # check compatibility if training is continued from previously saved gen_model
    if args.init_from is not None:
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No gen_model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["gen_model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved gen_model disagree on '%s' "%checkme
        
        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded gen_model disagreee on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded gen_model disagreee on dictionary mappings!"
        
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)
    
    disc_model = DiscModel(args)
    gen_model = GenModel(args)
    gen_model.attach_cost(disc_model)
    disc_model.attach_cost(gen_model)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("coolgraph", sess.graph)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore gen_model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        training_disc = False
        training_gen = True
        for e in range(args.num_epochs):
            sess.run(tf.assign(gen_model.lr, args.learning_rate * (args.decay_rate ** e)))
            sess.run(tf.assign(disc_model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                # TODO: WHY MUST WE FEED THE DISC MODEL HERE? 
                # I need to look into the TensorFlow code, but I think this is because in the disciminator model, I use the same
                # RNN chain for disciminating words that are fed from the text corpus and disciminating the generator's output. 
                # I think this chain sees that there are nodes in the graph which (although not necessary to be fed), could impact the output. 
                # Therefore it demands these nodes be fed even if they aren't actually used when the generator is run?
                gen_model_latent_state = np.random.uniform(-1., 1., (args.batch_size, args.latent_size)).astype('float32')
                gen_feed = {disc_model.input_data_text: np.zeros_like(x), gen_model.input_data: x, gen_model.latent_state: gen_model_latent_state}
                gen_outputs = [gen_model.loss, gen_model.outputs, gen_model.train_op, gen_model.embedding]

                disc_feed = {disc_model.input_data_text: x, gen_model.input_data: x, gen_model.latent_state: gen_model_latent_state}
                disc_outputs = [disc_model.loss, disc_model.train_op, disc_model.embedding]

                if training_gen:
                    gen_loss, gen_outputs, _, embedding = sess.run(gen_outputs, gen_feed)
                    print 'batch is {}, epoch is {}, gen_loss is {}'.format(e * data_loader.num_batches + b, e, gen_loss)
                    print 'gen output: '
                    print_wv_nn(embedding, gen_outputs, data_loader.vocab, args.batch_size)
                    if gen_loss < args.gen_train_bound:
                        training_gen = False
                        training_disc = True
                        print 'training the discriminator only...'
                if training_disc:
                    disc_loss, _, embedding = sess.run(disc_outputs, disc_feed)
                    print 'batch is {}, epoch is {}, disc_loss is {}'.format(e * data_loader.num_batches + b, e, disc_loss)
                    if disc_loss < args.disc_train_bound:
                        training_disc = False
                        training_gen = True
                        print 'training the generator only...'

                end = time.time()

                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
    main()
