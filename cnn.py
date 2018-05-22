from keras.datasets import imdb
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, Tokenizer
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
    item_max = 0
    item_counter = 0 
    for row in reader:
      if not first:
        for item in row:
          cols = item.split('\t')
          X.append(cols[2])
          y.append(int(cols[1]))
          item_counter += 1
      first = False
      if (item_counter > item_max):
        item_max = item_counter
        item_counter = 0
    print(item_max)
  return X, y

def oneHotEncode(X, y, corpus_size = None):
  X = [one_hot(d, corpus_size) for d in X]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
  return X_train, X_test, y_train, y_test

def tfidfEncode(X_train, X_test, corpus_size = None):
  token = Tokenizer(num_words=corpus_size)
  token.fit_on_texts(texts = X)
  X_train = token.texts_to_matrix(texts = X_train, mode='tfidf')
  X_test = token.texts_to_matrix(texts = X_test, mode='tfidf')
  return X_train, X_test

batch_size = 10
epochs = 10
corpus_size = 14547
nbr_entries = 4618
# 87 is the maximum length of a tweet
max_length = 87

X_train, X_test, y_train, y_test = readIrony()
X = X_train + X_test
y =  y_train + y_test

#X_train, X_test = tfidfEncode(X_train, X_test, corpus_size)
X_train, X_test, y_train, y_test = oneHotEncode(X, y, corpus_size)

#(X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = imdb.load_data(num_words=corpus_size)
#print(X_train_imdb)
np.set_printoptions(threshold=np.inf)
X = np.concatenate((X_train, X_test), axis = 0)
y = np.concatenate((y_train, y_test), axis = 0)
print ("X shape: ", )
print (np.shape(X))
print (np.shape(y))
print("About to print the entry in train set")
#print(X_train[1])
print("About to print the labels set")
#print(y)
print(len(np.unique(np.hstack(X))))
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length
#plt.boxplot(result)
#plt.show()
# We see the bulk of the data is within 250 words of length
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Start model creation
model = Sequential()
model.add(Embedding(corpus_size, 2, input_length=max_length))
model.add(Dropout(0.5))
model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu', activity_regularizer = regularizers.l1(0.00023)))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=0)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict(X_test)
for i in range(0, len(predictions)):
  if predictions[i] > 0.5:
    predictions[i] = 1
  else:
    predictions[i] = 0
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(classification_report(y_test, predictions))

exit(0)
top_words = 10000

print (X_train)
print()
print (X_test)
print()
print (y_train)
print()
print (y_test)