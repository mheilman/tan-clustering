#!/bin/env python3


import sys
import argparse
import glob
import re
import itertools
import math
from collections import defaultdict




def document_generator(path):
    with open(path) as f:
        for line in f.readlines():
            yield [x for x in line.strip().split() if x]

def test_doc_gen():
    for path in glob.glob('review_polarity/txt_sentoken/*/cv*'):
        with open(path) as f:
            sys.stderr.write('.')
            sys.stderr.flush()
            yield [x for x in re.split('\s+', f.read().lower()) if x]


class DocumentLevelClusters(object):
    def __init__(self, doc_generator, batch_size=1000):
        self.batch_size = batch_size
        self.cluster_parents = {}
        self.cluster_ids = {}
        self.cluster_counter = 0
        self.index = defaultdict(set)  # word ID -> list of doc IDs
        self.create_index(doc_generator)
        self.cluster_bits = {}
        self.word_bitstrings = {x: None for x in self.cluster_ids.keys()}  # word/cluster labels to bitstrings (initialize to include all words as keys)
        
        word_clusters = self.current_clusters()
        while len(word_clusters) > 1:
            # find the best to merge
            c1, c2 = self.find_pair_to_merge()

            # make pointers
            self.cluster_parents[c1] = self.cluster_counter
            self.cluster_parents[c2] = self.cluster_counter
            self.cluster_bits[c1] = '0'
            self.cluster_bits[c2] = '1'

            # merge the clusters in the index
            self.merge(c1, c2, self.cluster_counter)

            # increment the cluster index counter
            self.cluster_counter += 1

            word_clusters = self.current_clusters()
            print('MERGED:\t{}\t{}'.format(c1, c2), file=sys.stderr)

        self.create_bitstrings()

    def create_bitstrings(self):
        for w in self.word_bitstrings:
            # walk up the tree until there is no parent cluster
            cur_cluster = self.cluster_ids[w]
            while cur_cluster in self.cluster_parents:
                bitstring += self.cluster_bits[cur_cluster]
                cur_cluster = self.cluster_parents[cur_cluster]

            self.word_bitstrings[w] = bitstring

    def find_pair_to_merge(self):
        max_score = float('-inf')
        argmax_score = None

        for c1, c2 in itertools.combinations(self.index.keys(), 2):
            count_c1_c2 = self.count_pair(c1, c2)
            if count_c1_c2 == 0:
                continue
            score = math.log(count_c1_c2) - math.log(self.count(c1)) - math.log(self.count(c2))
            if score > max_score:
                max_score = score
                argmax_score = (c1, c2)
        return argmax_score

    def merge(self, c1, c2, new_id):
        self.index[new_id] = self.index[c1] | self.index[c2]
        del self.index[c1]
        del self.index[c2]

    def create_index(self, doc_generator):
        for doc_id, doc in enumerate(doc_generator):
            for w in set(doc):
                if w not in self.cluster_ids:
                    self.cluster_ids[w] = self.cluster_counter
                    self.cluster_counter += 1
                self.index[self.cluster_ids[w]].add(doc_id)


    def current_clusters(self):
        return self.index.keys()

    def count(self, c1):
        return len(self.index[c1])

    def count_pair(self, c1, c2):
        return len(self.index[c1] & self.index[c2])


def main():
    parser = argparse.ArgumentParser(description='Create hierarchical word clusters from a corpus follow Brown et al. (1992).')
    parser.add_argument('input_path', help='input file, one document per line, with whitespace-separated tokens.')
    args = parser.parse_args()

    c = DocumentLevelClusters(args.input_path)


if __name__ == '__main__':
    # main()
    c = DocumentLevelClusters(test_doc_gen())
    import pdb;pdb.set_trace()
