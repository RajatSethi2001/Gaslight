import numpy as np
import tensorflow as tf

from GradientEngine import gradientRun
from tensorflow.keras import models


def predict(image, extra=None):
    # tf.keras.backend.clear_session()
    # gc.collect()
    victim = extra["model"]
    image_input = image.reshape((1,) + image.shape)
    image_output = np.argmax(list(victim(image_input).numpy()[0]))
    return image_output

victim = models.load_model('Classifiers/mnist')

extra = {"model": victim}

input_shape = (28, 28, 1)

input_range = (0, 1)

target = None

model_name = "Models/PPO_50_params.zip"

param_file = "Params/more_params.pkl"

save_interval = 1000

gradientRun(predict, extra, input_shape, input_range, target, model_name, param_file, save_interval)
