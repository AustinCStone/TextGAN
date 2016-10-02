# -*- coding: utf-8 -*-
import os
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools


def print_wv_nn(embedding, approx_wv_outputs, vocab, batch_size):
    inv_vocab = {v: k for k, v in vocab.items()}
    normed_embedding = np.copy(embedding)
    avg_similarity = 0.
    for i, row in enumerate(normed_embedding):
        normed_embedding[i] = row / np.sqrt(max(sum(row**2), 1e-12))
    for i in range(batch_size):
        words_produced = []
        for output in approx_wv_outputs:
            # just print the first one in the batch for now
            o = output[i]
            dot = np.dot(normed_embedding, o)
            best_match = np.argmax(dot)
            avg_similarity += dot[best_match]
            words_produced.append(inv_vocab[best_match])
        print ' '.join(words_produced)
    print 'avg similarity is {}'.format(avg_similarity / (batch_size * len(approx_wv_outputs)))


def print_softmax(softmax_outputs, vocab, batch_size, seq_length):
    inv_vocab = {v: k for k, v in vocab.items()}
    for i in range(batch_size - seq_length):
        words_produced = []
        for output in softmax_outputs[i:i + seq_length]:
            best_match = np.argmax(output)
            words_produced.append(inv_vocab[best_match])
        print ' '.join(words_produced)



# TODO: fix this... It sucks
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, vocab_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`.]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = collections.Counter(sentences)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common(self.vocab_size)]
        vocabulary_inv = list(sorted(vocabulary_inv))
        vocabulary_inv = ['UNK', 'PAD', 'END'] + vocabulary_inv
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()

        # Optional text cleaning or make them lower case, etc.
        data = self.clean_str(data)
        x_text = data.split()
        self.vocab, self.words = self.build_vocab(x_text)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        self.tensor = []
        for word in x_text:
            if not self.vocab.has_key(word):
                self.tensor.append(self.vocab['UNK'])
            else:
                self.tensor.append(self.vocab[word])
        self.tensor = np.asarray(self.tensor)
        # Save the data to data.npy
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
