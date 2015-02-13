__author__ = 'miljan'

# from gensim.models.word2vec import Word2Vec
# from nltk.corpus import movie_reviews as mr
# from collections import defaultdict
# from nltk.tokenize import word_tokenize
# from pprint import pprint
# from os import chdir
#
# model = Word2Vec.load_word2vec_format('../Data/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
# print model['computer']
#
# documents = defaultdict(list)
# for i in mr.fileids():
#     documents[i.split('/')[0]].append(i)
# pprint(documents['pos'][:10]) # first ten pos reviews.
# pprint(documents['neg'][:10]) # first ten neg reviews.
#
# datapath = '/Users/miljan/nltk_data/corpora/movie_reviews/'
# sentences = []
# for id in mr.fileids():
#     with open(datapath + id) as file:
#         lines = file.readlines()
#         for line in lines:
#             sentences.append(word_tokenize(line))
#
# model.train(sentences)



import numpy as np
import pickle
### Set parameters ###

np.random.seed(42)
validation_size = 0.2

### Construct train, validation and test dataset ###

rawdata = np.loadtxt(open("../Data/test2.csv", "rb"), delimiter=",")
# shuffle rows for randomness
# np.random.shuffle(rawdata)

cutoff = int(round(len(rawdata) * (1 - validation_size)))

labels_train = rawdata[:cutoff, 0]
labels_validate = rawdata[cutoff:-1, 0]
labels_test = rawdata[-1, 0]

words = np.delete(rawdata, 0, 1)
words_train = words[:cutoff, :]
words_validate = words[cutoff:-1, :]
words_test = words[-1, :]

data_train = (words_train, labels_train)
data_validate = (words_validate, labels_validate)
data_test = (words_test, labels_test)

data_all = (data_train, data_validate, data_test)
pickle.dump(data_all, open("theano.p", "wb"))