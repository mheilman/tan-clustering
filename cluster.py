#!/bin/env python3


import sys
import argparse
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
import glob
import re
import itertools
from sqlalchemy import func, distinct
import math


Base = declarative_base()

Session = sessionmaker()
engine = create_engine('sqlite:///:memory:', echo=False)
Session.configure(bind=engine)


class IndexEntry(Base):
    __tablename__ = 'IndexEntry'

    id = Column(Integer, primary_key=True)
    cluster = Column(Integer)
    doc_id = Column(Integer)

    def __init__(self, cluster, doc_id):
        self.cluster = cluster
        self.doc_id = doc_id

    def __repr__(self):
        return "{} {}".format(self.cluster, self.doc_id)



Base.metadata.create_all(engine)




def document_generator(path):
    with open(path) as f:
        for line in f.readlines():
            yield [x for x in line.strip().split() if x]

def test_doc_gen():
    for path in glob.glob('review_polarity/txt_sentoken/*/cv*')[:50]:
        with open(path) as f:
            sys.stderr.write('.')
            sys.stderr.flush()
            yield [x for x in re.split('\s+', f.read()) if x]


class DocumentLevelClusters(object):
    def __init__(self, doc_generator, batch_size=100):
        self.batch_size = batch_size
        self.session = Session()
        self.cluster_parents = {}
        self.cluster_ids = {}
        self.cluster_counter = 0
        self.index_documents(doc_generator)
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
            self.relabel(c1, self.cluster_counter)
            self.relabel(c2, self.cluster_counter)

            # increment the cluster index counter
            self.cluster_counter += 1

            word_clusters = self.current_clusters()
            print('MERGING:\t{}\t{}'.format(c1, c2), file=sys.stderr)

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

        logcounts = {x[0]: math.log(x[1]) for x in self.session.query(IndexEntry.cluster, func.count()).group_by(IndexEntry.cluster).order_by(func.count().desc()).limit(self.batch_size).all()}

        for c1, c2 in itertools.combinations(logcounts, 2):
            count_c1_c2 = self.count_pair(c1, c2)
            if count_c1_c2 == 0:
                continue
            score = math.log(count_c1_c2) - logcounts[c1] - logcounts[c2]
            if score > max_score:
                max_score = score
                argmax_score = (c1, c2)
        return argmax_score

    def relabel(self, old_id, new_id):
        for entry in self.session.query(IndexEntry).filter(IndexEntry.cluster == old_id).all():
            entry.word = new_id
        self.session.commit()

    def index_documents(self, doc_generator):
        for doc_id, doc in enumerate(doc_generator):
            for w in set(doc):
                if w not in self.cluster_ids:
                    self.cluster_ids[w] = self.cluster_counter
                    self.cluster_counter += 1
                self.session.add(IndexEntry(self.cluster_ids[w], doc_id))
            self.session.commit()

        #print("# the = {}".format(self.session.query(IndexEntry).filter(IndexEntry.cluster == 'the').count()), file=sys.stderr)

    def current_clusters(self):
        return [x[0] for x in self.session.query(distinct(IndexEntry.cluster)).all()]

    def count(self, c1):
        return self.session.query(IndexEntry.id).filter(IndexEntry.cluster == c1).count()

    def count_pair(self, c1, c2):
        return self.session.query(IndexEntry.id).filter(IndexEntry.cluster == c1).union(self.session.query(IndexEntry.id).filter(IndexEntry.cluster == c2)).count()


def main():
    parser = argparse.ArgumentParser(description='Create hierarchical word clusters from a corpus follow Brown et al. (1992).')
    parser.add_argument('input_path', help='input file, one document per line, with whitespace-separated tokens.')
    args = parser.parse_args()

    c = DocumentLevelClusters(args.input_path)


if __name__ == '__main__':
    # main()
    c = DocumentLevelClusters(test_doc_gen())
    import pdb;pdb.set_trace()
