
import numpy
from nltk.util import ngrams

#
def kmeansSentimentWords(clean_sent, centroids):

    pass


# derive ngram bagOfWords representation of a review
# ! NOTE: ngram_vocab should be ngram_vocab = defaultdict(int)
def buildBagOfNgrams(clean_sent, ngram_vocab, n):
    l_ngrams = ngrams(clean_sent, n)
    for ngram in l_ngrams:
        ngram_vocab[ngram] += 1
    return ngram_vocab

# derive sentence representation have sum of word vectors
def buildSentVecAsSum(clean_sent, model):

    temp = numpy.zeros((1,300))

    for w in clean_sent:
        try:
            temp = temp + model[w]
        except:
            pass

    return temp


# derive sentence representation as average of word vectors
def buildSentVecAsAverage(clean_sent, model):

    temp = numpy.zeros((1,300))
    N = 0

    for w in clean_sent:
        try:
            temp = temp + model[w]
            N = N+1
        except:
            pass
    if N>0:
        temp = temp/N

    return temp
