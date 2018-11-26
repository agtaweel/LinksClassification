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
from sklearn.preprocessing import Imputer

nltk.download('stopwords')
filename = 'finalized_model.sav'


# Importing the dataset


def read_data(path):
    print('reading data')
    dataset = pd.read_csv(path)
    imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    dataset['class'] = imputer.fit_transform(dataset[['class']]).ravel()
    x = dataset.iloc[:, :-1].values
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
        print((i + 1) / len(x) * 100, '%')
        print(i+2)
    x = bag_of_words(corpus)
    y = dataset.iloc[:, 1].values
    # Splitting the dataset into the Training set and Test set
    print('splitting data')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    print('data was read')
    return x_train, x_test, y_train, y_test, x


# Creating Bag of Words model
def bag_of_words(corpus):
    cv = CountVectorizer(max_features=1500)
    x = cv.fit_transform(corpus).toarray()
    return x


# Get link data


def open_link(link):
        f = requests.get(link)
        page = BeautifulSoup(f.content, 'html.parser')
        return page


def train_data(x_train, y_train):

    print('start training data')
    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    print('fitting data')
    classifier.fit(x_train, y_train)

    # save the model to disk
    pickle.dump(classifier, open(filename, 'wb'))
    print('training finished')
    return classifier


def get_results(classifier, x_test, y_test):
    # Predicting the Test set results
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    score = f1_score(y_test,predictions)
    print('testing results: ')
    return score, accuracy


if __name__ == "__main__":
    x_train, x_test, y_train, y_test,x = read_data('Dataset.csv')
    classifier = train_data(x_train, y_train)
    score, accuracy = get_results(classifier, x_test, y_test)
    print('score = ', score*100, '%')
    print('accuracy = ', accuracy*100, '%')
