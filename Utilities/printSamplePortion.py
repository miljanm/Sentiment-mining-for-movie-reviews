
#imports
from csv import reader
import os

# user params
relative_path = '../Data/text'
limit = 10

# initialize path
dir = os.path.dirname(__file__)
abs_path = os.path.join(dir, relative_path)

# open file
print '\n------------------'
print 'Open File...'
file = open(abs_path)
reader = reader(file, dialect="excel-tab")
reader.next()

# loop over lines
i = 0
for line in reader:

    # show line
    print line

    # early exit
    if i>limit:
        break