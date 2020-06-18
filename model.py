from tensorflow.keras import model_from_json
import numpy as np
import tensorflow as tf

config = tf.config.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compt.v1.Session(config = config)


class FacialExpressionModel(object):

    EMOTION_LIST = [
    "Angry", "Disgust",
    "Fear", "Happy", "Neutral",
    "Sad", "Surprise"
    ]

    def __init__(self, json_file, weights_file):
        with open(json_file, "r") as file:
            loaded_model = file.read()
            self.loaded = model_from_json(loaded_model)

        self.loaded_model.load_weights(weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTION_LIST[np.argmax(self.preds)]
