import tensorflow as tf
from tensorflow import keras
import numpy as np

# Charger les données
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normaliser les données
X_train = X_train/255
X_test = X_test/255

# Créer le modèle
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid")
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test)
model.save("./saved_model/")
