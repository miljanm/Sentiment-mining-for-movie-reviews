
import numpy

"""

test = numpy.zeros((3, 2, 2))

test[(0,0,0)]=1
test[(0,0,1)]=2
test[(0,1,0)]=2
test[(0,1,1)]=0
test[(1,0,0)]=1
test[(1,0,1)]=1
test[(1,1,0)]=1
test[(1,1,1)]=0
test[(2,0,0)]=3
test[(2,0,1)]=5
test[(2,1,0)]=4
test[(2,1,1)]=3

print test

print '\n'
print 'sum...'
print '\n'

print numpy.sum(test, axis=0)

"""

"""
a = numpy.array([[1,2,1],[4,2,6],[8,1,4],[3,4,1]])

print a
print numpy.argmax(a, axis=0)
print numpy.argmax(a, axis=1)
"""


###########
# Imports #
###########
from nltk.corpus import stopwords
from nltk import word_tokenize

cachedRemovalList = stopwords.words("english")
cachedRemovalList = cachedRemovalList+["'",",","."]

sent = "how are movie car you doin '"

print [word for word in word_tokenize(sent.lower()) if word not in cachedRemovalList]


