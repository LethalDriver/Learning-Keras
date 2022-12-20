from tensorflow import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

INPUT_SIZE = 28*28

train_images = train_images.reshape((60000, INPUT_SIZE))
train_images = train_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

test_images = test_images.reshape((10000, INPUT_SIZE))
test_images = test_images.astype("float32") / 255

network = models.Sequential()
network.add(layers.Dense(512, input_shape=(INPUT_SIZE, ), activation="relu"))
network.add(layers.Dense(10, activation="softmax"))
network.compile(optimizer="rmsprop",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_acc:", test_acc)

