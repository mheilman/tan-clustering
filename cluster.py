#!/bin/env python3

import json
import sys
import argparse
import glob
import re
import itertools
#import math
from collections import defaultdict
from time import gmtime, strftime
import numpy

def curtimestr():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

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
    def __init__(self, doc_generator, batch_size=1000):
        self.batch_size = batch_size
        self.cluster_parents = {}
        self.cluster_counter = 0
        self.index = defaultdict(set)  # word ID -> list of doc IDs
        self.word_bitstrings = {}  # word/cluster labels to bitstrings (initialize to include all words as keys)
        self.create_index(doc_generator)
        self.cluster_bits = {}
        self.log_counts = {}
        
        # initialize the dictionary of classes to consider merging and their log counts
        most_common_words = sorted(self.index.keys(), key=lambda x: -len(self.index[x]))
        self.current_batch = most_common_words[:self.batch_size]
        self.current_batch_scores = [(self.log_count_pair(c1, c2) - self.log_count(c1) - self.log_count(c2), (c1, c2)) for c1, c2 in itertools.combinations(self.current_batch, 2)]

        most_common_words = most_common_words[self.batch_size:]

        print('\n{} CLUSTERING'.format(curtimestr()), file=sys.stderr)
        while len(self.index) > 1:
            # find the best to merge
            c1, c2 = self.find_best(self.current_batch_scores)
            
            # pick the new cluster ID and increment the cluster index counter
            new_cluster = 'C\t{}'.format(self.cluster_counter)

            # merge the clusters in the index
            self.merge(c1, c2, new_cluster)

            # remove the merged clusters from the batch, add the new one and the next most frequent word (if available)
            self.update_batch(c1, c2, new_cluster, most_common_words)

            print('{}\t{} AND {} WERE MERGED INTO {}. {} REMAIN.'.format(curtimestr(), c1, c2, new_cluster, len(self.index)), file=sys.stderr)

            self.cluster_counter += 1

        self.create_bitstrings()

    def update_batch(self, c1, c2, new_cluster, most_common_words):
        self.current_batch = [x for x in self.current_batch if x != c1 and x != c2]
        self.current_batch_scores = [x for x in self.current_batch_scores if not (x[1][0] is c1 or x[1][1] is c1 or x[1][0] is c2 or x[1][1] != c2)]
        # TODO might be able to make this list comprehension faster by using integer IDs
        
        new_items = [new_cluster]
        if most_common_words:
            new_word = most_common_words.pop(0)
            new_items.append(new_word)

        self.current_batch_scores.extend((self.log_count_pair(n1, n2) - self.log_count(n1) - self.log_count(n2), (n1, n2)) for n1, n2 in itertools.product(new_items, self.current_batch))
        self.current_batch.extend(new_items)

    def find_best(self, current_batch_scores):
        best_score, (c1, c2) = current_batch_scores[0]
        for score, (tmp1, tmp2) in current_batch_scores:
            if score > best_score:
                best_score = score
                c1, c2 = tmp1, tmp2
        return c1, c2

    def create_bitstrings(self):
        for w in self.word_bitstrings:
            # walk up the tree until there is no parent cluster
            cur_cluster = w
            bitstring = ""
            while cur_cluster in self.cluster_parents:
                bitstring += self.cluster_bits[cur_cluster]
                cur_cluster = self.cluster_parents[cur_cluster]

            self.word_bitstrings[w] = bitstring

    def log_count(self, c1):
        if c1 not in self.log_counts:
            self.log_counts[c1] = numpy.log(len(self.index[c1]))
        return self.log_counts[c1]

    def log_count_pair(self, c1, c2):
        count = len(self.index[c1] & self.index[c2])
        if count == 0:
            return float('-inf')
        return numpy.log(count)

    def find_pair_to_merge(self, current_batch):
        max_score = float('-inf')
        argmax_score = None
        
        count_c1_c2 = 0
        score = 0
        for (c1, log_count_c1), (c2, log_count_c2) in itertools.combinations(current_batch.items(), 2):
            count_c1_c2 = len(self.index[c1] & self.index[c2])
            if count_c1_c2 == 0:
                continue
            score = numpy.log(count_c1_c2) - log_count_c1 - log_count_c2
            if score > max_score:
                max_score = score
                argmax_score = (c1, c2)
        return argmax_score

    def merge(self, c1, c2, new_id):
        self.cluster_parents[c1] = new_id
        self.cluster_parents[c2] = new_id
        self.cluster_bits[c1] = '0'
        self.cluster_bits[c2] = '1'

        self.index[new_id] = self.index[c1] | self.index[c2]
        del self.index[c1]
        del self.index[c2]

    def create_index(self, doc_generator):
        for doc_id, doc in enumerate(doc_generator):
            for w in set(doc):
                self.word_bitstrings[w] = None
                self.index[w].add(doc_id)


def main():
    parser = argparse.ArgumentParser(description='Create hierarchical word clusters from a corpus follow Brown et al. (1992).')
    parser.add_argument('input_path', help='input file, one document per line, with whitespace-separated tokens.')
    parser.add_argument('output_path', help='output path')
    args = parser.parse_args()

    c = DocumentLevelClusters(args.input_path)

    with open(args.output_path, 'w') as f:
        json.dump(c.word_bitstrings, f)

if __name__ == '__main__':
    # main()
    c = DocumentLevelClusters(test_doc_gen())

    with open('bitstrings.json', 'w') as f:
        json.dump(c.word_bitstrings, f)
