
###########
# Imports #
###########
import time
import numpy as np
import pickle, gzip

##################
# Initialization #
##################
np.random.seed(42)
validation_size = 0.000667

################
# Loading Data #
################
print '\n------------------'
print 'Reading train and test vectors'

start_time = time.time()

rawTrainData = np.loadtxt(open("../Data/transformedData.csv", "rb"), delimiter=",")
rawTestData = np.loadtxt(open("../Data/transformedTestData.csv", "rb"), delimiter=",")
np.random.shuffle(rawTrainData)

loadData_time = time.time() - start_time

#############################
# Build TRAIN-VAL-TEST Sets #
#############################
print '\n------------------'
print 'Creating data structures ...'

start_time = time.time()

cutoff1 = int(round(len(rawTrainData) * (1 - validation_size)))

labels_train = rawTrainData[:cutoff1, 0].astype(int)
labels_validate = rawTrainData[cutoff1:, 0].astype(int)
labels_test = rawTestData[:, 0].astype(int)

rawTrainData[:, 0] = np.transpose(np.ones((rawTrainData.shape[0])))
rawTestData[:, 0] = np.transpose(np.ones((rawTestData.shape[0])))
words_train = rawTrainData[:cutoff1, :]
words_validate = rawTrainData[cutoff1:, :]
words_test = rawTestData[:, :]

data_train = (words_train, labels_train)
data_validate = (words_validate, labels_validate)
data_test = (words_test, labels_test)

create_Structures_time = time.time() - start_time

#####################
# PICKLE for THEANO #
#####################
print '\n------------------'
print 'Pickling ...'

start_time = time.time()

data_all = (data_train, data_validate, data_test)
pickle.dump(data_all, open("../Data/mnist.pkl", "wb"))

pickle_time = time.time() - start_time

#############
# Profiling #
#############
print '\n------------------'
print "Profiling...\n"
print "Loading: %f seconds" % loadData_time
print "Creating data structures: %f seconds" % create_Structures_time
print "Pickling: %f seconds" % pickle_time

##########
# Ending #
##########
print '\n------------------'
print 'Done!'