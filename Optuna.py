import numpy as np

from ParamFinder import ParamFinder
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

eps = 1

target = None

model_name = "Models/PPO_Reward.zip"

framework = "PPO"

param_file = "Params/PPO_Reward.pkl"

#How many trials to run for this iteration.
trials = 20

samples = 20

#How many timesteps to run through per sample.
timesteps = 2000

mode = "reward"

if __name__=='__main__':
    param_finder = ParamFinder(predict, extra, input_shape, input_range, eps, target, model_name, framework, param_file, trials, samples, timesteps, mode)
    param_finder.run()


