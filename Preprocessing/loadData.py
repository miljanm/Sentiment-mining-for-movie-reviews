
# imports
from csv import reader
import time
from gensim.models.word2vec import Word2Vec
from gensim.corpora import WikiCorpus

# utility variables
i=1
limit=1000000
trainset={}

# word2vec parameters
num_features = 300
min_word_count = 2
num_workers = 4
context = 10
downsampling = 1e-3

# paths
model_name = "../TrainedModels/word_vect_model"
train_path = "../Data/train.tsv"

# load data
print '\n------------------'
print 'Loading competition data...'
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

# loading wikipedia corpus
# print '\n------------------'
# print "Loading additional data..."
# corpus = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2',dictionary=True)

# train word2vec on the training set
print '\n------------------'
print "Training model..."
start_time = time.time()

sentences = []
for key, val in trainset.iteritems():
    sentences.append([key,val[1]])

model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# max_sentence = -1
# def generate_lines():
#     for index, text in enumerate(corpus.get_texts()):
#         if index < max_sentence or max_sentence==-1:
#             yield text
#         else:
#             break
# model = Word2Vec()
# model.build_vocab(generate_lines())
# model.train(generate_lines(),chunksize=500)

running_time = time.time() - start_time

# make more mem efficient and store
model.init_sims(replace=True)
model.save(model_name)

# test
print '\n------------------'
print "Testing semantic relations...\n"
# print model.most_similar("movie")
# print model.doesnt_match("france england germany berlin".split())
# print model.most_similar("man")

# random prints
print '\n------------------'
print "Printing data...\n"
print "dictionary length: %d" % len(trainset)
print "sentence length: %d" % len(sentences)
print "Running time: %f seconds" % running_time

# ending
print '\n------------------'
print "Ending..."