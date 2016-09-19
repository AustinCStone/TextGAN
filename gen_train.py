from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from gen_model import GenModel
from disc_model import DiscModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--gen_model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--disc_model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')                       
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved gen_model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to gen_model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'gen_model.ckpt-*'      : file(s) with gen_model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    
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
    gen_model = GenModel(args, disc_model)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore gen_model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(gen_model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            #state = gen_model.initial_state.eval()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                gen_feed = {gen_model.input_data: x, gen_model.initial_state: np.zeros((args.batch_size, 4*args.rnn_size))}
                gen_outputs = sess.run([gen_model.outputs], gen_feed)
                print (len(gen_outputs))
                print (len(gen_outputs[0]))
                outputs_wv = np.asarray(gen_outputs[0]).transpose(1, 0, 2)
                disc_feed = {disc_model.input_data_wv: outputs_wv, disc_model.targets: np.random.randint(2, size=args.batch_size).astype('float32').reshape((50, 1))}
                disc_train_loss, gen_train_loss, _, _ = sess.run([disc_model.cost_wv, gen_model.cost, disc_model.train_op_wv, gen_model.train_op], disc_feed)
                end = time.time()
                print("{}/{} (epoch {}), disc_train_loss = {:.3f}, gen_train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, disc_train_loss, gen_train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'gen_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("gen_model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
