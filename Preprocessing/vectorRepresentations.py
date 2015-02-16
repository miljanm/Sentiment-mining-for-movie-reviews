
import numpy

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

    temp = temp/N

    return temp
