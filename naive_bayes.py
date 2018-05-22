import numpy as np
import sys
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report


def readIrony(): 
  X_train, y_train = readData('SemEval2018-Task3-master/datasets/train/SemEval2018-T3-train-taskA.txt')
  X_test, y_test = readData('SemEval2018-Task3-master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt')
  return X_train, X_test, y_train, y_test

def readData(file):
  X = []
  y = []
  with open(file, 'r') as f:
    reader = csv.reader(f, delimiter='\n')
    first = True
    for row in reader:
      if not first:
        for item in row:
          cols = item.split('\t')
          X.append(cols[2])
          y.append(int(cols[1]))
      first = False
  return X, y
  

def trainNB(X_train, y_train):
  #X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.1, random_state=37)
  nb = MultinomialNB()
  print ("Training")
  model = nb.fit(X_train, y_train)
  return model

def countVectorize(documents):
  print("Vectorising corpus by count")
  cv = CountVectorizer()
  return cv.fit_transform(documents)

def tfidfVectorize(documents):
  print("Vectorising corpus by tfidf")
  tfidf = TfidfVectorizer()
  return tfidf.fit_transform(documents)

def classify_nb(model , documentTokens):
  print("Classifying")
  return model.predict(documentTokens)

def main():
  X_train, X_test, y_train, y_test = readIrony()
  X = X_train + X_test
  tfidf_vectorizer = TfidfVectorizer()
  tfidf_vectorizer.fit(X_train)
  X_train = tfidf_vectorizer.transform(X_train).toarray()
  X_test = tfidf_vectorizer.transform(X_test).toarray()
  
  model = trainNB(X_train, y_train)
  predictions = classify_nb(model, X_test) 
  accuracy = accuracy_score(y_test, predictions)
  precision = precision_score(y_test, predictions)
  recall = recall_score(y_test, predictions)
  f1 = f1_score(y_test, predictions)
  print(classification_report(y_test, predictions))
  print("Accuracy: %s \nPrecision score: %s \nRecall Score: %s \nF1 Score: %s" % (accuracy, precision, recall, f1))

main()
exit(0)

