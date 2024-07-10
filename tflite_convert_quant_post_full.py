import random

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Utilisé pour créer l'ensemble des données et les charger par lot
class Dataset:
    def __init__(self, data, label):
        # les chemins des images
        self.data = data
        # les chemins des images de segmentation
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # print("Build model")
        # lire les données
        return self.data[i], self.label[i]

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = size

    def __getitem__(self, i):
        # collecter les données du lot
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return self.size // self.batch_size

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

train_dataset = Dataset(X_train, y_train)
train_loader = Dataloader(train_dataset, 1, len(train_dataset))

def representative_data_gen():
    for idx in range(len(train_loader)):
        data = train_loader.__getitem__(idx)
        yield [np.array(data[0], dtype=np.float32, ndmin=2)]

converter = tf.lite.TFLiteConverter.from_saved_model("./saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_quant_full_model = converter.convert()

print(len(tflite_quant_full_model))

with open("tflite_quant_full_model.tflite", "wb") as f:
    f.write(tflite_quant_full_model)
