import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_model_optimization as tfopt

# Charger les données
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normaliser les données
X_train = X_train/255
X_test = X_test/255

# Charger le modèle
keras_model = tf.keras.models.load_model("saved_model.h5")
keras_model.summary()

# Créer le modèle quantifié
q_keras_model  = tfopt.quantization.keras.quantize_model(keras_model)
q_keras_model.compile( optimizer="adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

q_keras_model.summary()

q_keras_model.fit(X_train, y_train, epochs=1)
q_keras_model.evaluate(X_test, y_test)

# Convertir en TFLite

converter = tf.lite.TFLiteConverter.from_keras_model(q_keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_qaware_model = converter.convert()

print(len(tflite_qaware_model))
with open("tflite_qaware_model.tflite", 'wb') as f:
    f.write(tflite_qaware_model)
