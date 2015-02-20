
###########
# Imports #
###########
from nltk import word_tokenize
from nltk.stem.snowball import *


# clean sentence, input string, default output string
def cleanItUp(sent, removeList, stringed=True, stemmed=True):

    list = [word for word in word_tokenize(sent.lower()) if word not in removeList]

    if stemmed:
        list = __stemItUp(list)

    return ''.join(list) if stringed else list


# stem words
def __stemItUp(list):

    stemmer = SnowballStemmer("english")
    return map(lambda x: stemmer.stem(x), list)
