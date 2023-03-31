import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import keras_preprocessing.text
import pandas as pd
import numpy as np
import json
import matplotlib as plt
from matplotlib import pyplot

df = pd.read_csv('Tweets.csv');

df = df.drop(columns=['textID', 'selected_text'])

data = df['text']
labels = df['sentiment']

labels = np.unique(labels, return_inverse=True)
lookup = labels[0]
labels = labels[1]

data = np.array(data).astype(str)
tokenizer = keras_preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
one_hot_results = tokenizer.texts_to_matrix(data, mode='binary')

all_tweets = one_hot_results[:20000]
all_labels = labels[:20000]

train_data = all_tweets[:10000]
test_data = all_tweets[10000:]


train_labels = all_labels[:10000]
test_labels = all_labels[10000:]


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(test_data, test_labels, batch_size=512, epochs=20)
history_dict = history.history

test_loss, test_acc = model.evaluate(train_data, train_labels)
print("test_acc:", test_acc)

