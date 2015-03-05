
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
neighborWeight = 0.27 #0.2
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

test_samples = sum(1 for line in open(fileList[0]))-1

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
            if not singleVote:
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
    #results = numpy.argmax(summed_votes, axis = 1)

    # Optimal set: Samr_Theano
    # Optimal weights: 0.265, 0.205, 0.06, 0.205, 0.255
    # NB: DECREASE LAST 2 MAKES WORSE, DECREASE FIRST 2 (maybe 3) MAKES BETTER
    # TRY ADDING ANOTHER SAMR

    weights = numpy.tile(numpy.array([0.265, 0.205, 0.06, 0.205, 0.255]), (summed_votes.shape[0], 1))
    labels = numpy.tile(numpy.array([0,1,2,3,4]), (summed_votes.shape[0], 1))
    temp1 = numpy.multiply(summed_votes, weights)
    temp2 = numpy.multiply(temp1, labels)
    temp3 = numpy.multiply(summed_votes, weights)
    results = numpy.divide(numpy.sum(temp2, axis=1), numpy.sum(temp3, axis=1))

    for row in results.tolist():
        write_out.writerow([first_id+row_idx,int(round(row))])
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
