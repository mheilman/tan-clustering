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
    for path in glob.glob('review_polarity/txt_sentoken/*/cv*')[:5]:
        with open(path) as f:
            sys.stderr.write('.')
            sys.stderr.flush()
            yield [x for x in re.split('\s+', f.read().lower()) if x]


class DocumentLevelClusters(object):
    def __init__(self, doc_generator, batch_size=100):
        self.batch_size = batch_size
        self.cluster_parents = {}
        self.cluster_ids = {}
        self.cluster_counter = 0
        self.index = defaultdict(set)  # word ID -> list of doc IDs
        self.create_index(doc_generator)
        self.cluster_bits = {}
        self.word_bitstrings = {x: None for x in self.cluster_ids.keys()}  # word/cluster labels to bitstrings (initialize to include all words as keys)
        
        # initialize the dictionary of classes to consider merging and their log counts
        most_common_words = sorted(self.index.keys(), key=lambda x: -len(self.index[x]))
        current_batch = {x: self.log_count(x) for x in most_common_words[:self.batch_size]}
        most_common_words = most_common_words[self.batch_size:]

        print('\n{} CLUSTERING'.format(curtimestr()), file=sys.stderr)
        word_clusters = self.current_clusters()
        while len(word_clusters) > 1:
            # find the best to merge
            c1, c2 = self.find_pair_to_merge(current_batch)

            # pick the new cluster ID and increment the cluster index counter
            new_cluster = self.cluster_counter

            # merge the clusters in the index
            self.merge(c1, c2, new_cluster)

            # remove the merged clusters from the batch, add the new one and the next most frequent word (if available)
            self.update_batch(c1, c2, new_cluster, current_batch, most_common_words)
            
            word_clusters = self.current_clusters()
            print('{}\t{} AND {} WERE MERGED INTO {}. {} REMAIN.'.format(curtimestr(), c1, c2, new_cluster, len(word_clusters)), file=sys.stderr)

            self.cluster_counter += 1

        self.create_bitstrings()

    def update_batch(self, c1, c2, new_cluster, current_batch, most_common_words):
        del current_batch[c1]
        del current_batch[c2]
        current_batch[new_cluster] = self.log_count(new_cluster)
        if most_common_words:
            new_word = most_common_words.pop(0)
            current_batch[new_word] = self.log_count(new_word)

    def create_bitstrings(self):
        for w in self.word_bitstrings:
            # walk up the tree until there is no parent cluster
            cur_cluster = self.cluster_ids[w]
            bitstring = ""
            while cur_cluster in self.cluster_parents:
                bitstring += self.cluster_bits[cur_cluster]
                cur_cluster = self.cluster_parents[cur_cluster]

            self.word_bitstrings[w] = bitstring

    def log_count(self, c1):
        return numpy.log(len(self.index[c1]))

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
                if w not in self.cluster_ids:
                    self.cluster_ids[w] = self.cluster_counter
                    self.cluster_counter += 1
                self.index[self.cluster_ids[w]].add(doc_id)


    def current_clusters(self):
        return self.index.keys()


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
    import pdb;pdb.set_trace()

    with open('bitstrings.json', 'w') as f:
        json.dump(c.word_bitstrings, f)
