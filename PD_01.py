from keras import datasets
data = datasets.mnist.load_data()
(x_train,y_train),(x_test,y_test) = data
print(x_train.shape)
from keras import models
from keras import layers
import numpy as np

input_shape = (28,28)
def createModel(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
model = createModel(input_shape)
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

def genModel(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(63, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
 
input_shape = (28,28,1)
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1) 
print(x_train.shape)    
model = genModel(input_shape)
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

