import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def save_tflite_model(model, model_path, input_gen):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.representative_dataset = input_gen

    tflite_quant_model = converter.convert()

    os.makedirs(model_path, exist_ok=True)

    with open(f"{model_path}/tflite_model.tflite", "wb") as file:
        file.write(tflite_quant_model)


def load_tflite_model(model_path):
    with tfmot.quantization.keras.quantize_scope():
        loaded_model = tf.keras.models.load_model(model_path)

    return loaded_model
