
import numpy


# derive simple bagOfWords representation of a review
def buildBagOfWords():

    #vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    #train_data_features = vectorizer.fit_transform(clean_reviews)
    print 'not yet implemented'


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
