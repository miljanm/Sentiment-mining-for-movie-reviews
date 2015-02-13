
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def testFuncOld():
    text = 'hello bye the the hi'
    text = [word for word in text.split() if word not in stopwords.words("english")]
    print text

def testFuncNew():
    text = 'hello bye the the hi'
    text = [word for word in text.split() if word not in cachedStopWords]
    print text

if __name__ == "__main__":

    testFuncOld()
    testFuncNew()