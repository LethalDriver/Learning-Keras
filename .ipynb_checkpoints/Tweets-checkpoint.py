import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import keras_preprocessing.text
import pandas as pd
import numpy as np

df = pd.read_csv('Tweets.csv');

df = df.drop(columns=['textID', 'text'])

data = df['selected_text']
labels = df['sentiment']

labels = np.unique(labels, return_inverse=True)
lookup = labels[0]
labels = labels[1]

data = np.array(data).astype(str)
tokenizer = keras_preprocessing.text.Tokenizer(num_words=5000)
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
model.add(layers.Dense(64, activation='relu', input_shape=(5000, )))
model.add(keras.layers.Dropout(0.4))
model.add(layers.Dense(16, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, batch_size=512, epochs=20, validation_split=0.3)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print("test_acc:", test_acc)

