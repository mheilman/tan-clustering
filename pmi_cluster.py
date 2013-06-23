#!/bin/env python3

'''
A little module for creating hierarchical word clusters.
This is based on the following paper.

Peter F. Brown; Peter V. deSouza; Robert L. Mercer; T. J. Watson; Vincent J.
Della Pietra; Jenifer C. Lai. 1992.  Class-Based n-gram Models of Natural
Language.  Computational Linguistics, Volume 18, Number 4.
http://acl.ldc.upenn.edu/J/J92/J92-4003.pdf


While this code creates hierarchical clusters, it does not use the HMM-like
sequence model to do so (section 3).  Instead, it merges clusters similar to the
technique described in section 4 of Brown et al. (1992), using pointwise mutual
information.  However, the formulation of PMI used here differs slightly.
Instead of using a window, we compute PMI using the probability that
two randomly selected clusters from the same document will be c1 and c2.
Also, since the total number of cluster tokens and pairs are constant,
we just use counts instead of probabilities.
Thus, the score for merging two clusters c1 and c2 is the following:

log[count(two tokens in the same doc are in c1 in c2) / count(c1) / count(c2)]

* See http://www.cs.columbia.edu/~cs4705/lectures/brown.pdf for a nice
  overview of Brown clustering.

* Here is another implementation of Brown clustering:
  https://github.com/percyliang/brown-cluster

* Also, see Percy Liang's Master's Thesis:
  Percy Liang. 2005.  Semi-supervised learning for natural language.  MIT.
  http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf

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

random.seed(1234567890)
from math import log

logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')


def document_generator(path):
    with open(path) as f:
        #for line in f.readlines():
        #    yield [x for x in line.strip().split() if x]
        paragraphs = [x for x in re.split(r'\n+', f.read()) if x]
        for paragraph in paragraphs:
            yield [x for x in re.split(r'\W+', paragraph.lower()) if x]


def test_doc_gen_reviews():
    for path in glob.glob('review_polarity/txt_sentoken/*/cv*'):
        with open(path) as f:
            yield re.split(r'\s+', f.read().strip().lower())
            # sys.stderr.write('.')
            # sys.stderr.flush()
            # for line in f.readlines():
            #     yield [x for x in re.split('\s+', line.strip().lower()) if x]


def test_doc_gen():
    docs = ['dog cat bird bat whale monkey',
            'monkey human ape',
            'human man woman child',
            'fish whale shark',
            'man woman teacher lawyer doctor',
            'fish shark',
            'bird bat fly']
    return map(str.split, docs)


class DocumentLevelClusters(object):
    '''
    The initializer takes a document generator, which is simply an iterator
    over lists of tokens.  You can define this however you wish.
    '''
    def __init__(self, doc_generator, batch_size=1000, max_vocab_size=None):
        self.batch_size = batch_size
        self.num_docs = 0

        self.max_vocab_size = max_vocab_size

        # mapping from cluster IDs to cluster IDs,
        # to keep track of the hierarchy
        self.cluster_parents = {}
        self.cluster_counter = 0

        # cluster_id -> {doc_id -> counts}
        self.index = defaultdict(dict)

        # the list of words in the vocabulary and their counts
        self.words = []
        self.word_counts = defaultdict(int)

        # the 0/1 bit to add when walking up the hierarchy
        # from a word to the top-level cluster
        self.cluster_bits = {}

        # create sets of documents that each word appears in
        self.create_index(doc_generator)

        # find the most frequent words
        # apply document count threshold.
        # include up to max_vocab_size words (or fewer if there are ties).
        self.create_vocab()

        # make a copy of the list of words, as a queue for making new clusters
        word_queue = list(self.words)

        # score potential clusters, starting with the most frequent words.
        # also, remove the batch from the queue
        self.current_batch = word_queue[:(self.batch_size + 1)]
        self.current_batch_scores = list(self.make_pair_scores(itertools.combinations(self.current_batch, 2)))
        word_queue = word_queue[(self.batch_size + 1):]

        while len(self.current_batch) > 1:
            # find the best pair of words/clusters to merge
            c1, c2 = self.find_best()

            # merge the clusters in the index
            self.merge(c1, c2)

            # remove the merged clusters from the batch, add the new one
            # and the next most frequent word (if available)
            self.update_batch(c1, c2, word_queue)

            logging.info('{} AND {} WERE MERGED INTO {}. {} REMAIN.'
                         .format(c1, c2, self.cluster_counter,
                                 len(self.current_batch) + len(word_queue) - 1))

            self.cluster_counter += 1

    def create_index(self, doc_generator):
        for doc_id, doc in enumerate(doc_generator):
            for w in doc:
                if doc_id not in self.index[w]:
                    self.index[w][doc_id] = 0
                self.index[w][doc_id] += 1
                self.word_counts[w] += 1

        # just add 1 to the last doc id (enumerate starts at zero)
        self.num_docs = doc_id + 1
        logging.info('{} documents were indexed.'.format(self.num_docs))

    def create_vocab(self):
        self.words = sorted(self.word_counts.keys(),
                            key=lambda w: self.word_counts[w], reverse=True)

        if self.max_vocab_size is not None \
           and len(self.words) > self.max_vocab_size:
            too_rare = self.word_counts[self.words[self.max_vocab_size + 1]]
            if too_rare == self.word_counts[self.words[0]]:
                too_rare += 1
                logging.info("max_vocab_size too low.  Using all words that" +
                             " appeared >= {} times.".format(too_rare))

            self.words = [w for w in self.words
                          if self.word_counts[w] > too_rare]
            words_set = set(self.words)
            index_keys = list(self.index.keys())
            for key in index_keys:
                if key not in words_set:
                    del self.index[key]
                    del self.word_counts[key]

    def make_pair_scores(self, pair_iter):
        for c1, c2 in pair_iter:
            paircount = 0
            # call set() on the keys for compatibility with python 2.7 and pypy
            for doc_id in (set(self.index[c1].keys()) & set(self.index[c2].keys())):
                paircount += self.index[c1][doc_id] * self.index[c2][doc_id]

            if paircount == 0:
                yield (float('-inf'), (c1, c2))  # log(0)
                continue

            score = log(paircount) \
                    - log(self.word_counts[c1]) \
                    - log(self.word_counts[c2])

            yield (score, (c1, c2))

    def find_best(self):
        best_score, (c1, c2) = self.current_batch_scores[0]
        for score, (tmp1, tmp2) in self.current_batch_scores:
            # break ties randomly (randint takes inclusive args!)
            if score > best_score \
               or (score == best_score and random.randint(0, 1) == 1):
                best_score = score
                c1, c2 = tmp1, tmp2
        return c1, c2

    def merge(self, c1, c2):
        c_new = self.cluster_counter

        self.cluster_parents[c1] = c_new
        self.cluster_parents[c2] = c_new
        r = random.randint(0, 1)
        self.cluster_bits[c1] = str(r)  # assign bits randomly
        self.cluster_bits[c2] = str(1 - r)

        # initialize the document counts of the new cluster with the counts
        # for one of the two child clusters.  then, add the counts from the
        # other child cluster
        self.index[c_new] = self.index[c1]
        for doc_id in self.index[c2]:
            if doc_id not in self.index[c_new]:
                self.index[c_new][doc_id] = 0
            self.index[c_new][doc_id] += self.index[c2][doc_id]

        # sum the frequencies of the child clusters
        self.word_counts[c_new] = self.word_counts[c1] + self.word_counts[c2]

        # remove merged clusters from the index to save memory
        # (but keep frequencies for words for the final output)
        del self.index[c1]
        del self.index[c2]
        if c1 not in self.words:
            del self.word_counts[c1]
        if c2 not in self.words:
            del self.word_counts[c2]

    def update_batch(self, c1, c2, freq_words):
        # remove the clusters that were merged (and the scored pairs for them)
        self.current_batch = [x for x in self.current_batch
                              if not (x == c1 or x == c2)]
        self.current_batch_scores = [x for x in self.current_batch_scores
                                     if not (x[1][0] == c1 or x[1][1] == c1
                                             or x[1][0] == c2 or x[1][1] == c2)]

        # find what to add to the current batch
        new_items = [self.cluster_counter]
        if freq_words:
            new_word = freq_words.pop(0)
            new_items.append(new_word)

        # add to the batch and score the new cluster pairs that result
        self.current_batch_scores.extend(self.make_pair_scores(itertools.product(new_items, self.current_batch)))
        self.current_batch_scores.extend(self.make_pair_scores(itertools.combinations(new_items, 2)))

        # note: make the scores first with itertools.product
        # (before adding new_items to current_batch) to avoid duplicates
        self.current_batch.extend(new_items)

    def get_bitstring(self, w):
        # walk up the cluster hierarchy until there is no parent cluster
        cur_cluster = w
        bitstring = ""
        import pdb;pdb.set_trace()
        while cur_cluster in self.cluster_parents:
            bitstring = self.cluster_bits[cur_cluster] + bitstring
            cur_cluster = self.cluster_parents[cur_cluster]
        return bitstring

    def save_clusters(self, output_path):
        with open(output_path, 'w') as f:
            for w in self.words:
                f.write("{}\t{}\t{}\n".format(w, self.get_bitstring(w),
                                          self.word_counts[w]))



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

    #doc_generator = document_generator(args.input_path)
    doc_generator = test_doc_gen_reviews()

    c = DocumentLevelClusters(doc_generator,
                              max_vocab_size=args.max_vocab_size,
                              batch_size=args.batch_size)
    c.save_clusters(args.output_path)


if __name__ == '__main__':
    main()
