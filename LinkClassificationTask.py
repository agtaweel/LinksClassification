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
from sklearn.metrics import accuracy_score, f1_score

nltk.download('stopwords')

# Importing the dataset
print('reading data')
dataset = pd.read_csv('Dataset.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
print('data was read')
# Get link data


def open_link(link):
    f = requests.get(link)
    page = BeautifulSoup(f.content, 'html.parser')
    return page


def train_data(x, dataset):
    print('start training data')

    corpus = []
    for i in range(0, len(x)):
        content = open_link(''.join(x[i]))
        page = re.sub('[^\u0627-\u064a]', ' ', str(content))
        page = page.lower()
        page = page.split()
        ps = PorterStemmer()
        page = [ps.stem(word) for word in page if not word in set(stopwords.words('arabic'))]
        page = ' '.join(page)
        corpus.append(page)
        print((i+1)/len(x)*100, '%')

    # Creating Bag of Words model
    cv = CountVectorizer(max_features=1500)
    x = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    print('splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    print('fitting data')
    classifier.fit(x_train, y_train)

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    print('training finished')
    return classifier, x_test, y_test


def get_results(classifier, x_test, y_test):
    # Predicting the Test set results
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    score = f1_score(y_test,predictions)
    print('testing results: ')
    return score, accuracy


if __name__ == "__main__":
    classifier, x_test, y_test = train_data(x, dataset)
    score, accuracy = get_results(classifier, x_test, y_test)
    print('score = ', score)
    print('accuracy = ', accuracy)
