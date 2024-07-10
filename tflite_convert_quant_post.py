
import tensorflow as tf

# Charger le modèle Keras
keras_model = tf.keras.models.load_model("saved_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model_quantize_f16 = converter.convert()


with open("tflite_model_quantize_f16.tflite", "wb") as f:
    f.write(tflite_model_quantize_f16)


