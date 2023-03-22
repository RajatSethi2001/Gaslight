import numpy as np
import tensorflow as tf

from GradientEngine import gradientRun
from tensorflow.keras import models


def predict(image, extra=None):
    victim = extra["model"]
    image_input = image.reshape((1,) + image.shape)
    image_output = np.argmax(list(victim(image_input).numpy()[0]))
    return image_output

victim = models.load_model('Classifiers/mnist')

extra = {"model": victim}

input_shape = (28, 28, 1)

input_range = (0, 1)

max_delta = 0.2

target = None

norm = 2

model_name = "Models/PPO_Medium.zip"

framework = "PPO"

save_interval = 1000

param_file = "Params/PPO_Medium.pkl"

gradientRun(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, save_interval, param_file)