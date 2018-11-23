# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Importing the dataset
dataset = pd.read_csv('Dataset.csv')
X = dataset.iloc[0:99, :-1].values
y = dataset.iloc[0:99, 1].values
# X = pd.DataFrame(X)

# Get link data


def open_link(link):
    f = requests.get(link)
    page = BeautifulSoup(f.content, 'html.parser')
    return page


corpus = []
data = []
for i in range(0, 100):
    content = open_link(''.join(X[i]))
    data.append(content)
    # print(str(content))
    page = re.sub('[^\u0627-\u064a]', ' ', str(content))
    # print(page)
    page = page.lower()
    page = page.split()
    ps = PorterStemmer()
    page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
    page = ' '.join(page)
    # print(page)
    corpus.append(page)

# Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:99, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
