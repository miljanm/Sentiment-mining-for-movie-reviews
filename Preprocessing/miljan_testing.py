__author__ = 'miljan'

from gensim.models.word2vec import Word2Vec
# from nltk.corpus import chat80

model = Word2Vec.load_word2vec_format('../Data/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format

print model.most_similar(positive=['woman', 'king'], negative=['man'])
print model.doesnt_match("breakfast cereal dinner lunch".split())
print model.similarity('woman', 'man')
print model['computer']  # raw numpy vector of a word
