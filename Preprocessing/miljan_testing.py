__author__ = 'miljan'

from gensim.models.word2vec import Word2Vec
from nltk.corpus import movie_reviews as mr
from collections import defaultdict
from nltk.tokenize import word_tokenize
from pprint import pprint
from os import chdir

model = Word2Vec.load_word2vec_format('../Data/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format

# documents = defaultdict(list)
# for i in mr.fileids():
#     documents[i.split('/')[0]].append(i)
# pprint(documents['pos'][:10]) # first ten pos reviews.
# pprint(documents['neg'][:10]) # first ten neg reviews.

datapath = '/Users/miljan/nltk_data/corpora/movie_reviews/'
sentences = []
for id in mr.fileids():
    with open(datapath + id) as file:
        lines = file.readlines()
        for line in lines:
            sentences.append(word_tokenize(line))

model.train(sentences)
