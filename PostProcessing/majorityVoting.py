
###########
# Imports #
###########
import csv
from os import walk
import numpy


##################
# Initialization #
##################
my_path = '../Data/Ensemble/'
singleVote = False
neighborWeight = 0.2

##############################
# Collect Models to Ensemble #
##############################
fileList = []

for (dirpath, dirnames, filenames) in walk(my_path):
    fileList.extend(filenames)
fileList = [dirpath+e for e in fileList]

csvins = [open(f, 'rb') for f in fileList]
reads = [csv.reader(handler, delimiter = ',') for handler in csvins]

test_samples = sum(1 for line in open(fileList[0]))


#################
# Extract Votes #
#################
reader_idx = 0
votes = numpy.zeros((test_samples, 5, len(fileList)))

for read in reads:
    line_idx = 0
    read.next()

    for row in read:
        pred = int(row[1])
        votes[line_idx, pred, reader_idx] = 1

        if ~singleVote:
            if pred+1<=4:
                votes[line_idx, pred+1, reader_idx] = neighborWeight
            if pred-1>=0:
                votes[line_idx, pred-1, reader_idx] = neighborWeight

        line_idx = line_idx + 1

    reader_idx = reader_idx + 1


#################
# Combine Votes #
#################
#with open('../Data/ensemplePredictions.csv', 'wb') as csvout:

#    write_out = csv.writer(csvout, delimiter = ',')
#    write_out.writerow(['PhraseId', 'Sentiment'])

"""
with open('ensemplePredictions.csv', 'wb') as csvout:

	writer = csv.writer(csvout, delimiter=',')
	with open('f1.csv', 'rb') as csvin1:
		with open('f2.csv', 'rb') as csvin2:
			with open('f3.csv', 'rb') as csvin3:
				with open('f4.csv', 'rb') as csvin4:

					reader1 = csv.reader(csvin1, delimiter=',')
					reader2 = csv.reader(csvin2, delimiter=',')
					reader3 = csv.reader(csvin3, delimiter=',')
					reader4 = csv.reader(csvin4, delimiter=',')
					reader1.next()
					reader2.next()
					reader3.next()
					reader4.next()
					writer.writerow('id,click')

					for row1 in reader1:
						row2 = reader2.next()
						row3 = reader3.next()
						row4 = reader4.next()
						row1[1] = str( (float(row1[1])*0.27+float(row2[1])*0.26+float(row3[1])*0.27+float(row4[1]))*0.20 )
			 			writer.writerow(row1)
                        """