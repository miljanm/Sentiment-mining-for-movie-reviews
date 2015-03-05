
###########
# Imports #
###########
import time
from csv import reader
import cPickle as pickle
import operator
from nltk.util import ngrams
from collections import defaultdict
from sentenceCleaning import cleanItUp
from nltk.corpus import stopwords


##############
# Initialize #
##############
n = 1
max_num = 270
ngram_dict = defaultdict(int)
freq_nGrams = []
train_path = '../Data/train.tsv'


#########################
# Counting nGrams #
#########################
print '\n------------------'
print 'Counting nGrams...'

start_time = time.time()

removalList = stopwords.words("english")
removalList = removalList + ["'",",",".","ca'","n't","'s","-lrb-","-rrb-","''", '``']

with open(train_path) as tsv:

    # initialize reader and skip header
    r = reader(tsv, dialect="excel-tab")
    r.next()

    # store train sentences labels
    for line in r:
        clean_sent = cleanItUp(line[2], removalList, False, False)
        ngrams_list = ngrams(clean_sent, 2)
        for ngram in ngrams_list:
            ngram_dict[ngram] += 1

counting_time = time.time() - start_time


#######################
# Chopping Unfrequent #
#######################
print '\n------------------'
print 'Chopping ...'

start_time = time.time()

sorted_nGrams = sorted(ngram_dict.items(), key=operator.itemgetter(1), reverse=True)

freq_nGrams = map(lambda x: x[0], sorted_nGrams[0:max_num])
print freq_nGrams

chopping_time = time.time() - start_time


############
# Pickling #
############
print '\n------------------'
print 'Pickling ...'

start_time = time.time()

pickle.dump(freq_nGrams, open("../Data/frequentNgrams.pkl", "wb"))

pickle_time = time.time() - start_time


#############
# Profiling #
#############
print '\n------------------'
print "Printing...\n"
print "Counting nGrams: %f seconds" % counting_time
print "Chopping: %f seconds" % chopping_time
print "Pickling frequent nGrams: %f seconds" % pickle_time


##########
# Ending #
##########
print '\n------------------'
print "Done!"