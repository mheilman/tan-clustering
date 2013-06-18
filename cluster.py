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
information.  However, the formulation of PMI used here differs slightly: 
each word/cluster is associated with a list of documents that it appears in 
(documents can be full texts, sentences, tweets, etc.).  
The score for merging two clusters c1 and c2 is proportional to the following:

  log [ p(c1 and c2 in document) / p(c1 in document) / p(c2 in document) ]

The probabilities are maximum likelihood estimates (e.g., number of documents
with c1 divided by the number of documents).  Since the total number of 
documents is constant, we just use the counts instead of relative frequencies.

Also, see http://www.cs.columbia.edu/~cs4705/lectures/brown.pdf for a nice 
overview.

Another implementation of "Brown clustering":
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
import numpy as np

np.random.seed(1234567890)

logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')

def document_generator(path):
    with open(path) as f:
        for line in f.readlines():
            yield [x for x in line.strip().split() if x]

def test_doc_gen():
    for path in glob.glob('review_polarity/txt_sentoken/*/cv*'):
        with open(path) as f:
            sys.stderr.write('.')
            sys.stderr.flush()
            for line in f.readlines():
                yield [x for x in re.split('\s+', line.strip().lower()) if x]


class DocumentLevelClusters(object):
    '''
    The initializer takes a document generator, which is simply an iterator 
    over lists of tokens.  You can define this however you wish.
    '''
    def __init__(self, doc_generator, batch_size=1000):
        self.batch_size = batch_size

        # mapping from cluster IDs to cluster IDs, 
        # to keep track of the hierarchy
        self.cluster_parents = {}
        self.cluster_counter = 0

        # word ID -> list of doc IDs
        self.index = defaultdict(set)

        # word/cluster labels to bitstrings
        # (initialize to include all words as keys)
        self.word_bitstrings = {}
        
        # the bit to add when walking up the hierarchy 
        # from a word to the top-level cluster
        self.cluster_bits = {}

        # cache of log counts for words/clusters
        self.log_counts = {}

        # create sets of documents that each word appears in
        self.create_index(doc_generator)
        
        # find the most frequent words, score potential clusters for the most 
        # frequent words.  Note that the count-based score is proportional to 
        # the PMI since the probabilities are divided by a constant for the
        # number of documents in the input.
        most_common_words = sorted(self.index.keys(), 
                                   key=lambda x: -len(self.index[x]))
        self.current_batch = most_common_words[:(self.batch_size + 1)]
        self.current_batch_scores = list(self.make_pair_scores(itertools.combinations(self.current_batch, 2)))

        # remove the first batch_size elements
        most_common_words = most_common_words[(self.batch_size + 1):]

        logging.info('CLUSTERING')
        while len(self.index) > 1:
            # find the best pair of words/clusters to merge
            c1, c2 = self.find_best()
            
            # pick the new cluster ID and increment the cluster index counter
            new_cluster = 'C\t{}'.format(self.cluster_counter)

            # merge the clusters in the index
            self.merge(c1, c2, new_cluster)

            # remove the merged clusters from the batch, add the new one
            # and the next most frequent word (if available)
            self.update_batch(c1, c2, new_cluster, most_common_words)

            logging.info('{} AND {} WERE MERGED INTO {}. {} REMAIN.'
                  .format(c1, c2, new_cluster, len(self.index)))
            self.cluster_counter += 1

        # walk up the hierarchy from each word to create its bitstring
        self.create_bitstrings()

    def update_batch(self, c1, c2, new_cluster, most_common_words):
        # remove the clusters that were merged (and the scored pairs for them)
        self.current_batch = [x for x in self.current_batch 
                              if x != c1 and x != c2]
        self.current_batch_scores = [x for x in self.current_batch_scores 
                                     if not (x[1][0] is c1 or x[1][1] is c1 
                                             or x[1][0] is c2 or x[1][1] is c2)]
        
        # find what to add to the current batch
        new_items = [new_cluster]
        if most_common_words:
            new_word = most_common_words.pop(0)
            new_items.append(new_word)

        # add to the batch and score the new cluster pairs that result
        self.current_batch_scores.extend(self.make_pair_scores(itertools.product(new_items, self.current_batch)))

        # note: make the scores first with itertools.product 
        # (before adding new_items to current_batch) to avoid duplicates
        self.current_batch.extend(new_items)

    def make_pair_scores(self, pair_iter):
        for c1, c2 in pair_iter:
            paircount = len(self.index[c1] & self.index[c2])
            if paircount == 0:
                yield (float('-inf'), (c1, c1))  # log(0)
                continue
            if c1 not in self.log_counts:
                self.log_counts[c1] = np.log(len(self.index[c1]))
            if c2 not in self.log_counts:
                self.log_counts[c2] = np.log(len(self.index[c2]))

            yield (np.log(paircount) 
                  - self.log_counts[c1] 
                  - self.log_counts[c2],
                  (c1, c2))

    def find_best(self):
        best_score, (c1, c2) = self.current_batch_scores[0]
        for score, (tmp1, tmp2) in self.current_batch_scores:
            if score > best_score:
                best_score = score
                c1, c2 = tmp1, tmp2
        return c1, c2

    def create_bitstrings(self):
        for w in self.word_bitstrings:
            # walk up the cluster hierarchy until there is no parent cluster
            cur_cluster = w
            bitstring = ""
            while cur_cluster in self.cluster_parents:
                bitstring += self.cluster_bits[cur_cluster]
                cur_cluster = self.cluster_parents[cur_cluster]

            self.word_bitstrings[w] = bitstring

    def merge(self, c1, c2, new_id):
        self.cluster_parents[c1] = new_id
        self.cluster_parents[c2] = new_id
        r = np.random.randint(0, 1)
        self.cluster_bits[c1] = str(r)  # assign the bits arbitrarily
        self.cluster_bits[c2] = str(1 - r)

        self.index[new_id] = self.index[c1] | self.index[c2]
        del self.index[c1]
        del self.index[c2]

    def create_index(self, doc_generator):
        for doc_id, doc in enumerate(doc_generator):
            for w in set(doc):
                self.word_bitstrings[w] = None
                self.index[w].add(doc_id)


def main():
    parser = argparse.ArgumentParser(description='Create hierarchical word' +
        ' clusters from a corpus, following Brown et al. (1992).')
    parser.add_argument('input_path', help='input file, one document per' +
        ' line, with whitespace-separated tokens.')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    c = DocumentLevelClusters(args.input_path)

    with open(args.output_path, 'w') as f:
        for w, bitstring in c.word_bitstrings.items():
            print("{}\t{}".format(w, bitstring), file=f)

if __name__ == '__main__':
    # main()
    c = DocumentLevelClusters(test_doc_gen())

    with open('bitstrings.txt', 'w') as f:
        for w, bitstring in c.word_bitstrings.items():
            print("{}\t{}".format(w, bitstring), file=f)
