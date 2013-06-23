#!/bin/env python3

'''
A little module for creating hierarchical word clusters.
This is based on the following papers.

* Peter F. Brown; Peter V. deSouza; Robert L. Mercer; T. J. Watson; Vincent J.
  Della Pietra; Jenifer C. Lai. 1992.  Class-Based n-gram Models of Natural
  Language.  Computational Linguistics, Volume 18, Number 4.
  http://acl.ldc.upenn.edu/J/J92/J92-4003.pdf

* Percy Liang. 2005.  Semi-supervised learning for natural language.  MIT.
  http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf


Some additional references:

* See http://www.cs.columbia.edu/~cs4705/lectures/brown.pdf for a high-level
  overview of Brown clustering.

* Here is another implementation of Brown clustering:
  https://github.com/percyliang/brown-cluster


Author: Michael Heilman

'''

import random
import sys
import argparse
import glob
import re
import itertools
import logging
from collections import defaultdict
from time import gmtime, strftime
import random
from math import log, isnan, isinf

random.seed(1234567890)

logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')


def document_generator(path):
    #TODO
    with open(path) as f:
        #for line in f.readlines():
        #    yield [x for x in line.strip().split() if x]
        paragraphs = [x for x in re.split(r'\n+', f.read()) if x]
        for paragraph in paragraphs:
            yield [x for x in re.split(r'\W+', paragraph.lower()) if x]


def test_reviews():
    corpus = []
    for path in glob.glob('review_polarity/txt_sentoken/*/cv*'):
        with open(path) as f:
            corpus += [x for x in re.split(r'\s+', f.read()) if x]
    return corpus


def make_float_defaultdict():
    return defaultdict(float)


class DocumentLevelClusters(object):
    '''
    The initializer takes a document generator, which is simply an iterator
    over lists of tokens.  You can define this however you wish.
    '''
    def __init__(self, corpus, batch_size=1000, max_vocab_size=None):
        self.oov_id = -1
        self.batch_size = batch_size

        self.max_vocab_size = max_vocab_size

        # mapping from cluster IDs to cluster IDs,
        # to keep track of the hierarchy
        self.cluster_parents = {}
        self.cluster_counter = 0

        # the list of words in the vocabulary and their counts
        # (use floats for everything. seems to be faster)
        self.words = []
        self.counts = defaultdict(float)
        self.trans = defaultdict(make_float_defaultdict)
        self.num_tokens = 0.0

        # the graph weights (w) and the effects of merging nodes (L)
        # (see Liang's thesis)
        self.w = defaultdict(make_float_defaultdict)
        self.L = defaultdict(make_float_defaultdict)

        # the 0/1 bit to add when walking up the hierarchy
        # from a word to the top-level cluster
        self.cluster_bits = {}

        # create sets of documents that each word appears in
        self.create_index(corpus)

        # find the most frequent words
        # apply document count threshold.
        # include up to max_vocab_size words (or fewer if there are ties).
        self.create_vocab()

        # make a copy of the list of words, as a queue for making new clusters
        word_queue = list(self.words)

        # score potential clusters, starting with the most frequent words.
        # also, remove the batch from the queue
        self.current_batch = word_queue[:(self.batch_size + 1)]
        word_queue = word_queue[(self.batch_size + 1):]
        self.initialize_tables()

        while len(self.current_batch) > 1:
            # find the best pair of words/clusters to merge
            c1, c2 = self.find_best()

            # merge the clusters in the index
            self.merge(c1, c2)

            if word_queue:
                new_word = word_queue.pop(0)
                self.add_to_batch(new_word)

            logging.info('{} AND {} WERE MERGED INTO {}. {} REMAIN.'
                         .format(c1, c2, self.cluster_counter,
                                 len(self.current_batch) + len(word_queue) - 1))

            self.cluster_counter += 1

    def create_index(self, corpus):
        self.num_tokens = 0

        for w1, w2 in zip(corpus, corpus[1:]):
            self.trans[w1][w2] += 1.0
            self.counts[w1] += 1.0
            self.num_tokens += 1.0
        self.counts[w2] += 1.0
        self.num_tokens = self.num_tokens
        # note that these are all ints, and they will be used
        # in division operations, which won't work in python 2

        logging.info('{} word tokens were processed.'.format(self.num_tokens))

    def create_vocab(self):
        self.words = sorted(self.counts.keys(),
                            key=lambda w: self.counts[w], reverse=True)

        if self.max_vocab_size is not None \
           and len(self.words) > self.max_vocab_size:
            too_rare = self.counts[self.words[self.max_vocab_size + 1]]
            if too_rare == self.counts[self.words[0]]:
                too_rare += 1
                logging.info("max_vocab_size too low.  Using all words that" +
                             " appeared >= {} times.".format(too_rare))

            oov_words = set([w for w in self.words
                             if self.counts[w] <= too_rare])
            self.words = [w for w in self.words
                          if self.counts[w] > too_rare]

            for w in oov_words:
                # merge OOV counts
                self.counts[self.oov_id] += self.counts[w]
                del self.counts[w]

                # merge oov words in the "from" part of the transitions
                for w2, val in self.trans[w].items():
                    self.trans[self.oov_id][w2] += val
                del self.trans[w]

            # merge oov words in the "to" part of the transitions
            for w1 in self.trans:
                for w2 in set(self.trans[w1].keys()) & oov_words:
                    self.trans[w1][self.oov_id] += self.trans[w1][w2]
                    del self.trans[w1][w2]

    def initialize_tables(self):
        logging.info("initializing tables")
        trans = self.trans
        counts = self.counts

        # edges between nodes
        for c1, c2 in itertools.combinations(self.current_batch, 2):
            w = self.compute_weight(trans[c1][c2],
                                    trans[c2][c1],
                                    counts[c1],
                                    counts[c2])
            if w:
                self.w[c1][c2] = w

        # edges to and from a single node
        for c in self.current_batch:
            w = self.compute_weight(trans[c][c],
                                    None,
                                    counts[c],
                                    counts[c])
            if w:
                self.w[c][c] = w

        num_pairs = 0
        for c1, c2 in itertools.combinations(self.current_batch, 2):
            self.compute_L(c1, c2)
            num_pairs += 1
            if num_pairs % 1000 == 0:
                logging.info("{} pairs precomputed".format(num_pairs))

    def compute_weight(self, count_12, count_21, count_1, count_2):
        # equation 4.4 in Percy Liang's thesis
        res = 0.0
        n = self.num_tokens
        for paircount in (count_12, count_21):
            if paircount:
                res += (paircount / n) * log(paircount * n / count_1 / count_2)
        return res

    def compute_L(self, c1, c2):
        val = 0.0

        # avoid class member lookups
        # (this seems to give a decent performance gain since these are
        # called in loops over many items)
        count_c1 = self.counts[c1]
        count_c2 = self.counts[c2]
        trans = self.trans
        counts = self.counts
        compute_weight = self.compute_weight
        w = self.w

        # add the weight of edges coming in to the potential
        # new cluster from other nodes
        for d in self.current_batch:
            val += compute_weight(trans[c1][d] + trans[c2][d],
                                  trans[d][c1] + trans[d][c2],
                                  count_c1 + count_c2,
                                  counts[d])

        # ... but don't include what will be part of the new cluster
        for d in (c1, c2):
            val -= compute_weight(trans[c1][d] + trans[c2][d],
                                  trans[d][c1] + trans[d][c2],
                                  count_c1 + count_c2,
                                  counts[d])

        # add the weight of the edge from the potential new cluster
        # to itself
        val += compute_weight(trans[c1][c1] + trans[c1][c2]
                              + trans[c2][c1] + trans[c2][c2],
                              None,
                              count_c1 + count_c2,
                              count_c1 + count_c2)

        # subtract the weight of edges to/from c1, c2
        # (which would be removed)
        for d, c in itertools.product(self.current_batch, (c1, c2)):
            if d in w[c]:
                val -= w[c][d]
            elif c in w[d]:
                val -= w[d][c]

        self.L[c1][c2] = val

    def find_best(self):
        best_score = float('-inf')
        c1, c2 = None, None
        for tmp1 in self.L:
            for tmp2, score in self.L[tmp1].items():
                # break ties randomly
                if score > best_score \
                   or (score == best_score and random.randint(0, 2) == 1):
                    best_score = score
                    c1, c2 = tmp1, tmp2

        if isnan(best_score) or isinf(best_score):
            raise ValueError("bad value for score: {}".format(best_score))

        return c1, c2

    def merge(self, c1, c2):
        c_new = self.cluster_counter

        # avoid class member lookups (seems to help performance)
        trans = self.trans
        counts = self.counts
        compute_weight = self.compute_weight
        L = self.L
        w = self.w

        # record parents
        self.cluster_parents[c1] = c_new
        self.cluster_parents[c2] = c_new
        r = random.randint(0, 2)
        self.cluster_bits[c1] = str(r)  # assign bits randomly
        self.cluster_bits[c2] = str(1 - r)

        # add the new cluster to the counts and transitions dictionaries
        counts[c_new] = counts[c1] + counts[c2]
        for c in (c1, c2):
            for d, val in trans[c].items():
                if d == c1 or d == c2:
                    d = c_new
                trans[c_new][d] += val

        # update the score table
        for d1 in L:
            for d2 in L[d1]:
                for c in (c1, c2):
                    L[d1][d2] -= compute_weight(trans[d1][c] + trans[d2][c],
                                                trans[c][d1] + trans[c][d2],
                                                counts[d1] + counts[d2],
                                                counts[c])
                L[d1][d2] += compute_weight(trans[d1][c_new] + trans[d2][c_new],
                                            trans[c_new][d1] + trans[c_new][d2],
                                            counts[d1] + counts[d2],
                                            counts[c_new])

        # remove merged clusters from the counts and transitions dictionaries
        # to save memory (but keep frequencies for words for the final output)
        if c1 not in self.words:
            del counts[c1]
        if c2 not in self.words:
            del counts[c2]

        del trans[c1]
        del trans[c2]
        for d in trans:
            for c in [c1, c2]:
                if c in trans[d]:
                    del trans[d][c]

        # remove the old clusters from the w and L tables
        for table in (w, L):
            for d in table:
                if c1 in table[d]:
                    del table[d][c1]
                if c2 in table[d]:
                    del table[d][c2]
            if c1 in table:
                del table[c1]
            if c2 in table:
                del table[c2]

        # remove the merged items
        self.current_batch.remove(c1)
        self.current_batch.remove(c2)

        # add the new cluster to the w and L tables
        self.add_to_batch(c_new)

    def add_to_batch(self, c_new):
        # compute weights for edges connected to the new node
        for d in self.current_batch:
            self.w[d][c_new] = self.compute_weight(self.trans[d][c_new],
                                                   self.trans[c_new][d],
                                                   self.counts[d],
                                                   self.counts[c_new])
        self.w[c_new][c_new] = self.compute_weight(self.trans[c_new][c_new],
                                                   None,
                                                   self.counts[c_new],
                                                   self.counts[c_new])

        # add the new cluster and then compute scores for merging it
        # with all clusters in the current batch
        for d in self.current_batch:
            self.compute_L(d, c_new)

        self.current_batch.append(c_new)

    def get_bitstring(self, w):
        # walk up the cluster hierarchy until there is no parent cluster
        cur_cluster = w
        bitstring = ""
        while cur_cluster in self.cluster_parents:
            bitstring = self.cluster_bits[cur_cluster] + bitstring
            cur_cluster = self.cluster_parents[cur_cluster]
        return bitstring

    def save_clusters(self, output_path):
        with open(output_path, 'w') as f:
            for w in self.words:
                f.write("{}\t{}\t{}\n".format(w, self.get_bitstring(w),
                                          self.counts[w]))


def main():
    parser = argparse.ArgumentParser(description='Create hierarchical word' +
                                     ' clusters from a corpus, following' +
                                     ' Brown et al. (1992).')
    parser.add_argument('input_path', help='input file, one document per' +
                        ' line, with whitespace-separated tokens.')
    parser.add_argument('output_path', help='output path')
    parser.add_argument('--max_vocab_size', help='maximum number of words in' +
                        ' the vocabulary (a smaller number will be used if' +
                        ' there are ties at the specified level)',
                        default=None, type=int)
    parser.add_argument('--batch_size', help='number of clusters to merge at' +
                        ' one time (runtime is quadratic in this value)',
                        default=1000, type=int)
    args = parser.parse_args()

    #corpus = document_generator(args.input_path)
    corpus = test_reviews()

    #corpus = "the dog ran . the cat walked . the man ran . the child walked . a child spoke . a man walked . a man spoke . a dog ran .".split()
    c = DocumentLevelClusters(corpus,
                              max_vocab_size=args.max_vocab_size,
                              batch_size=args.batch_size)
    c.save_clusters(args.output_path)


if __name__ == '__main__':
    main()
