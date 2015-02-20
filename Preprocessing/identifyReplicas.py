
###########
# Imports #
###########
import time
from csv import reader
import cPickle as pickle


##############
# Initialize #
##############
trainset = {}
replicas = {}
train_path = '../Data/train.tsv'
test_path = '../Data/test.tsv'


#########################
# Load competition data #
#########################
print '\n------------------'
print 'Loading competition data...'

start_time = time.time()

with open(train_path) as tsv:

    # initialize reader and skip header
    r = reader(tsv, dialect="excel-tab")
    r.next()

    # store train sentences labels
    for line in r:
        trainset["".join(line[2].lower().split())] = line[3]

loading_time_train = time.time() - start_time
start_time = time.time()

with open(test_path) as tsv:

    # initialize reader and skip header
    r = reader(tsv, dialect="excel-tab")
    r.next()

    # loop over lines
    for line in r:

        if "".join(line[2].lower().split()) in trainset:
            replicas[line[0]] = trainset["".join(line[2].lower().split())]

loading_time_test = time.time() - start_time


############
# Pickling #
############
print '\n------------------'
print 'Pickling ...'

start_time = time.time()

pickle.dump(replicas, open("../Data/replicas.pkl", "wb"))

pickle_time = time.time() - start_time


#############
# Profiling #
#############
print '\n------------------'
print "Printing...\n"
print "Loading train sentences: %f seconds" % loading_time_train
print "Loading test sentences: %f seconds" % loading_time_test
print "Pickling replicas' dictionary: %f seconds" % pickle_time


##########
# Ending #
##########
print '\n------------------'
print "Done!"