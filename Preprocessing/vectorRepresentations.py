
import numpy
from nltk.util import ngrams

def kmeansSentiment(clean_sent, kMeansModel, pcaModel, word2vecModel):
    # initialize feature vector as dict
    params = kMeansModel.get_params()
    k = params['n_clusters']
    k_dict = dict.fromkeys(range(k), 0)

    for word in clean_sent.split():
        try:
            # read the word2vec vector and apply pca to it
            vector = pcaModel.transform(word2vecModel[word])
        except:
            continue
        prediction = kMeansModel.predict(vector)
        k_dict[prediction[0]] += 1
    k_dict_sorted = sorted(k_dict.items(), key=lambda x: x[0])
    return [c[1] for c in k_dict_sorted]


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


# if __name__ == '__main__':
#     from pickle import load
#     from gensim.models.word2vec import Word2Vec
#     kMeansModel = load(open("../Data/kMeans5.pkl", "rb" ))
#     pcaModel = load(open("../Data/pcaMLE.pkl", "rb" ))
#     model_name = '../Data/GoogleNews-vectors-negative300.bin'
#     word2vecModel = Word2Vec.load_word2vec_format(model_name, binary=True)  # C binary format
#     print kmeansSentiment('This movie is utter and complete rubbish', kMeansModel, pcaModel, word2vecModel)