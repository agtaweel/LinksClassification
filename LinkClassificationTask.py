# Data PreProcessing

# Importing the libraries
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nltk.download('stopwords')

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Get link data


def open_link(link):
    f = requests.get(link)
    page = BeautifulSoup(f.content, 'html.parser')
    return page


corpus = []
for i in range(0, len(X)):
    content = open_link(''.join(X[i]))
    page = re.sub('[^\u0627-\u064a]', ' ', str(content))
    page = page.lower()
    page = page.split()
    ps = PorterStemmer()
    page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
    page = ' '.join(page)
    corpus.append(page)

# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Predicting the Test set results
loaded_model = pickle.load(open(filename, 'rb'))
score = loaded_model.score(X_test, y_test)



