#!/bin/env python3

'''
A little module for creating hierarchical word clusters.
This is based loosely on the following paper.

Peter F. Brown; Peter V. deSouza; Robert L. Mercer; T. J. Watson; Vincent J.
Della Pietra; Jenifer C. Lai. 1992.  Class-Based n-gram Models of Natural
Language.  Computational Linguistics, Volume 18, Number 4.
http://acl.ldc.upenn.edu/J/J92/J92-4003.pdf


While this code creates hierarchical clusters, it does not use an HMM-like
sequence model to do so (Brown et al., 1992, section 3).  Instead, it merges
clusters simply by picking the pairs of clusters with the highest pointwise
mutual information. Instead of using a window (e.g., as in Brown et al., sec. 4),
this code computed PMI using the probability that two randomly selected clusters
from the same document will be c1 and c2.  Also, since the total numbers of
cluster tokens and pairs are constant across pairs, this code use counts
instead of probabilities. Thus, the score for merging two clusters
c1 and c2 is the following:

log[count(two tokens in the same doc are in c1 in c2) / count(c1) / count(c2)]

* See http://www.cs.columbia.edu/~cs4705/lectures/brown.pdf for a nice
  overview of Brown clustering.

* Here is another implementation of Brown clustering:
  https://github.com/percyliang/brown-cluster

* Also, see Percy Liang's Master's Thesis:
  Percy Liang. 2005.  Semi-supervised learning for natural language.  MIT.
  http://cs.stanford.edu/~pliang/papers/meng-thesis.pdf

Author: Michael Heilman (mheilman@ets.org, mheilman@cs.cmu.edu)

'''

import argparse
import glob
import re
import itertools
import logging
from collections import defaultdict
import random

random.seed(1234567890)
from math import log
from bs4 import UnicodeDammit

logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')


def document_generator(path, lower=False):
    '''
    Default document reader.  Takes a path to a file with one document per line,
    with tokens separate by whitespace, and yields lists of tokens per document.
    This could be replaced by any function that yields lists of tokens.
    See main() for how it is called.

    Note: this uses BeautifulSoup's UnicodeDammit to convert to unicode.
    '''
    with open(path, 'rb') as f:
        i = 0
        for line in f:
            line = UnicodeDammit(line.strip()).unicode_markup
            if line:
                if lower:
                    line = line.lower()
                i += 1
                if i % 100000 == 0:
                    logging.info('Read {} nonblank lines'.format(i))
                yield re.split(r'\s+', line)


def test_doc_gen():
    docs = ['dog cat bird bat whale monkey',
            'monkey human ape',
            'human man woman child',
            'fish whale shark',
            'man woman teacher lawyer doctor',
            'fish shark',
            'bird bat fly']
    return map(str.split, docs)


def make_float_defaultdict():
    return defaultdict(float)


def make_word_counts(document_generator, max_vocab_size=None, min_word_count=1):
    res = defaultdict(int)

    for doc in document_generator:
        for tok in doc:
            res[tok] += 1

    too_rare = min_word_count - 1
    if min_word_count > 1 and max_vocab_size is not None:
        logging.info("max_vocab_size and min_word_count both set." +
                     "  Ignoring min_word_count.".format(too_rare))

    words = sorted(res.keys(), key=lambda w: res[w], reverse=True)

    if max_vocab_size is not None:
        if len(words) <= max_vocab_size:
            too_rare = 0
        else:
            too_rare = res[words[max_vocab_size]]
            if too_rare == res[words[0]]:
                too_rare += 1
                logging.info("max_vocab_size too low.  Using all words" +
                             "that appeared > {} times.".format(too_rare))

    # only keep words that occur more frequently than too_rare
    words = [w for w in words if res[w] > too_rare]

    logging.info("Created vocabulary with the {} words that occurred at least {} times."
                 .format(len(words), too_rare + 1))

    words_set = set(words)
    wc_keys = list(res.keys())
    for key in wc_keys:
        if key not in words_set:
            del res[key]

    return res


class DocumentLevelClusters(object):
    '''
    Class for generating word clusters based on document-level co-occurence.
    The initializer takes a document generator, which is simply an iterator
    over lists of tokens.  You can define this however you wish.

    word_counts should be a dictionary of int counts that determines which words
    to consider.
    '''
    def __init__(self, doc_generator, word_counts, batch_size=1000):
        self.batch_size = batch_size
        self.num_docs = 0

        # mapping from cluster IDs to cluster IDs,
        # to keep track of the hierarchy
        self.cluster_parents = {}
        self.cluster_counter = 0

        # cluster_id -> {doc_id -> counts}
        self.index = defaultdict(dict)

        # the list of words in the vocabulary and their counts
        self.word_counts = word_counts

        # find the most frequent words
        self.words = sorted(self.word_counts.keys(),
                            key=lambda w: self.word_counts[w], reverse=True)

        # make a copy of the list of words, as a queue for making new clusters
        word_queue = list(self.words)

        # the 0/1 bit to add when walking up the hierarchy
        # from a word to the top-level cluster
        self.cluster_bits = {}

        # create sets of documents that each word appears in
        self.create_index(doc_generator)

        # score potential clusters, starting with the most frequent words.
        # also, remove the batch from the queue
        self.current_batch = word_queue[:(self.batch_size + 1)]
        self.current_batch_scores = defaultdict(make_float_defaultdict)
        self.make_pair_scores(itertools.combinations(self.current_batch, 2))
        word_queue = word_queue[(self.batch_size + 1):]
        while len(self.current_batch) > 1:
            # find the best pair of words/clusters to merge
            c1, c2 = self.find_best()

            # merge the clusters in the index
            self.merge(c1, c2)

            # remove the merged clusters from the batch, add the new one
            # and the next most frequent word (if available)
            self.update_batch(c1, c2, word_queue)

            logging.info(u'{} AND {} WERE MERGED INTO {}. {} REMAIN.'
                         .format(c1, c2, self.cluster_counter,
                                 len(self.current_batch) + len(word_queue) - 1))

            self.cluster_counter += 1

    def create_index(self, doc_generator):
        doc_id = 0
        for doc in doc_generator:
            for w in doc:
                if w not in self.word_counts:
                    continue
                if doc_id not in self.index[w]:
                    self.index[w][doc_id] = 0
                self.index[w][doc_id] += 1
            doc_id += 1

        self.num_docs = doc_id
        logging.info('{} documents were indexed.'.format(self.num_docs))

    def make_pair_scores(self, pair_iter):
        for c1, c2 in pair_iter:
            paircount = 0

            index1, index2 = self.index[c1], self.index[c2]
            if len(index1) > len(index2):
                index1, index2 = index2, index1

            for doc_id in index1:
                if doc_id not in index2:
                    continue
                paircount += index1[doc_id] * index2[doc_id]

            if paircount == 0:
                self.current_batch_scores[c1][c2] = float('-inf')  # log(0)
                continue

            # note that these counts are ints!
            # (but the log function returns floats)
            score = log(paircount) \
                    - log(self.word_counts[c1]) \
                    - log(self.word_counts[c2])

            self.current_batch_scores[c1][c2] = score

    def find_best(self):
        best_score = None
        argmax = None

        for c1, d in self.current_batch_scores.items():
            for c2, score in d.items():
                if best_score is None or score > best_score:
                    argmax = [(c1, c2)]
                    best_score = score
                elif score == best_score:
                    argmax.append((c1, c2))

        # break ties randomly (randint takes inclusive args!)
        c1, c2 = argmax[random.randint(0, len(argmax) - 1)]

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

        for c in [c1, c2]:
            if c in self.current_batch_scores:
                del self.current_batch_scores[c]
            for d in self.current_batch_scores.values():
                if c in d:
                    del d[c]

        # find what to add to the current batch
        new_items = [self.cluster_counter]
        if freq_words:
            new_word = freq_words.pop(0)
            new_items.append(new_word)

        # add to the batch and score the new cluster pairs that result
        self.make_pair_scores(itertools.product(new_items, self.current_batch))
        self.make_pair_scores(itertools.combinations(new_items, 2))

        # note: make the scores first with itertools.product
        # (before adding new_items to current_batch) to avoid duplicates
        self.current_batch.extend(new_items)

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
    parser.add_argument('--min_word_count', help='minimum word count to' +
                        'include a word in the vocabulary. (default: 1)',
                        default=1, type=int)
    parser.add_argument('--batch_size', help='number of clusters to merge at' +
                        ' one time (runtime is quadratic in this value)',
                        default=1000, type=int)
    parser.add_argument('--lower', help='lowercase the input',
                        action='store_true')
    args = parser.parse_args()

    word_counts = make_word_counts(document_generator(args.input_path, lower=args.lower),
                                   max_vocab_size=args.max_vocab_size,
                                   min_word_count=args.min_word_count)

    c = DocumentLevelClusters(document_generator(args.input_path, lower=args.lower),
                              word_counts, batch_size=args.batch_size)
    c.save_clusters(args.output_path)


if __name__ == '__main__':
    main()
