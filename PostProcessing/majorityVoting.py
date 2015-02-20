
###########
# Imports #
###########
import csv
from os import walk
import time
import numpy
import cPickle
import sys


##################
# Initialization #
##################
my_path = '../Data/Samr_Theano/'
pickle_path = '../Data/replicas.pkl'
checkReplicas = True
singleVote = False
neighborWeight = 0.5 #0.2
first_id = 156061

if checkReplicas:
    try:
        f = open(pickle_path, 'r')
        replicas = cPickle.load(f)
    except:
        print 'pickled replicas not found'
        sys.exit()


##############################
# Collect Models to Ensemble #
##############################
print '\n------------------'
print "Collecting...\n"
start_time = time.time()

fileList = []

for (dirpath, dirnames, filenames) in walk(my_path):
    fileList.extend(filenames)
fileList = [dirpath+e for e in fileList]

csvins = [open(f, 'rb') for f in fileList]
reads = [csv.reader(handler, delimiter = ',') for handler in csvins]

test_samples = sum(1 for line in open(fileList[0]))

collect_time = time.time() - start_time


#################
# Extract Votes #
#################
print '\n------------------'
print "Extracting...\n"
start_time = time.time()

reader_idx = 0
votes = numpy.zeros((test_samples, 5, len(fileList)))

for read in reads:
    line_idx = 0
    read.next()

    for row in read:
        sample_idx = row[0]
        pred = int(row[1])

        if sample_idx in replicas:
            votes[line_idx, replicas[sample_idx], reader_idx] = 1

        else:
            votes[line_idx, pred, reader_idx] = 1
            if ~singleVote:
                if pred+1<=4:
                    votes[line_idx, pred+1, reader_idx] = neighborWeight
                if pred-1>=0:
                    votes[line_idx, pred-1, reader_idx] = neighborWeight

        line_idx = line_idx + 1

    reader_idx = reader_idx + 1

extract_time = time.time() - start_time


#################
# Combine Votes #
#################
print '\n------------------'
print "Combining...\n"
start_time = time.time()
row_idx = 0

with open('../Data/ensemblePredictions.csv', 'wb') as csvout:

    write_out = csv.writer(csvout, delimiter = ',')
    write_out.writerow(['PhraseId', 'Sentiment'])

    summed_votes = numpy.sum(votes, axis = 2)
    majority_pred = numpy.argmax(summed_votes, axis = 1)

    for row in majority_pred.tolist():
        write_out.writerow([first_id+row_idx,row])
        row_idx = row_idx + 1

combine_time = time.time() - start_time


#############
# Profiling #
#############
print '\n------------------'
print "Profiling...\n"
print "Collecting models: %f seconds" % collect_time
print "Extracting votes: %f seconds" % extract_time
print "Combining votes: %f seconds" % combine_time


##########
# Ending #
##########
print '\n------------------'
print 'Done!'
