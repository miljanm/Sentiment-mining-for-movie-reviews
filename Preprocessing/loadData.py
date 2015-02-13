
# Imports
import time
from csv import reader, writer
from gensim.models.word2vec import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords

# Utility variables
i=1
limit=10
trainset={}

# Paths
model_name = '../Data/GoogleNews-vectors-negative300.bin'
train_path = "../Data/train.tsv"

# Load google pre-trained word vectors
print '\n------------------'
print 'Loading word2vec model...'

start_time = time.time()
model = Word2Vec.load_word2vec_format(model_name, binary=True)  # C binary format
model_time = time.time() - start_time

# Load competition data
print '\n------------------'
print 'Loading competition data...'

start_time = time.time()

with open(train_path) as tsv:
    # intitialize reader and skip header
    reader = reader(tsv, dialect="excel-tab")
    reader.next()

    # loop over lines
    for line in reader:

        # add to dictionary
        trainset[line[0]]=line[1:4]
        i=i+1

        # early exit
        if i>limit:
            break

reviewLoading_time = time.time() - start_time

# Vectorize sentences
print '\n------------------'
print 'Vectorizing sentences'

start_time = time.time()
cachedStopWords = stopwords.words("english")

with open('../transformedData.csv', 'wb') as csvout:
    my_writer = writer(csvout, delimiter=',')
    for i in range(1,len(trainset)):
        curr_sent = trainset[str(i)][1]
        for word in word_tokenize(curr_sent):
            my_writer.writerow(model[word])

sentVect_time = time.time() - start_time

# Profiling
print '\n------------------'
print "Printing...\n"
print trainset['1']
print "Loading word2vec model: %f seconds" % model_time
print "Loading movie reviews: %f seconds" % reviewLoading_time
print "Vectorizing sentences: %f seconds" % sentVect_time

# Ending
print '\n------------------'
print "Ending..."