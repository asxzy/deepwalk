#!/usr/bin/env python
import logging
import random
import sys
from collections import defaultdict
from gensim.models.word2vec import Word2Vec


logger = logging.getLogger("deepwalk")

class Graph():
    def __init__(self,num_paths=5,path_length=20,alpha=0.85):
        self.Graph = defaultdict(list)
        self.map = defaultdict(int)
        self.alpha = 0.85
        self.num_paths = num_paths
        self.path_length = path_length
        self.alpha = alpha
        return

    def load_edgelist(self, filename, undirected = True, mapit = True):
        with open(filename) as f:
            for index,line in enumerate(f):
                logger.info("%s lines loaded" % index)
                x, y = line.rstrip().split()
                if x not in self.map:
                    self.map[x] = len(self.map)
                if y not in self.map:
                    self.map[y] = len(self.map)
                self.Graph[self.map[x]].append(self.map[y])
                self.Graph[self.map[y]].append(self.map[x])
        self.map = dict((str(v),k) for k,v in self.map.iteritems())
        return

    def random_walk(self, path_length, start):
        G = self.Graph
        path = [start]

        while len(path) < path_length:
            cur = path[-1]
            if random.random() <= self.alpha:
                path.append(random.choice(G[cur]))
            else:
                path.append(path[0])
        return [self.map[str(x)] for x in path]

    def nodes(self):
        return self.Graph.keys()

    def __iter__(self):
        nodes = list(self.nodes())
        for cnt in range(self.num_paths):
            random.shuffle(nodes)
            for index,node in enumerate(nodes):
                if index % 100000 == 0:
                    logger.info("walking: %s / %s" % (index,cnt))
                yield self.random_walk(self.path_length, node)

    def vocab(self):
        return defaultdict(int,(zip([str(x) for x in self.Graph.keys()],[len(x) for x in self.Graph.values()])))


if __name__ == "__main__":
    G = Graph()
    G.load_edgelist(sys.argv[1])
    model = Word2Vec(min_count=0)
    model.raw_vocab = G.vocab()
    model.corpus_count = G.path_length*G.num_paths*len(G.Graph)
    model.finalize_vocab()
    model.train(G)
    model.save_word2vec_format('text')
