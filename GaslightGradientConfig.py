import numpy as np
import tensorflow as tf

from GradientEngine import gradientRun

#input_array - Array of values to be classified. Will have a shape of "input_shape" and range of "input_range"
#extra - Additional argument used to pass in any other values, like the classifier model.
#This function should return the top-1 output of the classifier. It can be any type, as long as "target" is a possible output and the type has a valid "==" operation.
def predict(input_array, extra=None):
    victim = extra["model"]
    input_reshaped = input_array.reshape((1,) + input_array.shape)
    label = np.argmax(list(victim(input_reshaped).numpy()[0]))
    return label

#Extra input for the predict function. Stores the classifier model so it doesn't have to be reloaded each iteration.
extra = {"model": tf.keras.models.load_model('Classifiers/mnist')}

#Shape of the classifier input.
input_shape = (28, 28, 1)

#Range of the inputs values. First value is min, second value is max. Preferably make this (0, 1) and scale in predict.
input_range = (0, 1)

#Maximum deviation per input parameter in a single action.
max_delta = 0.2

#Target label for attacking model to achieve. Corresponds with output from predict(). Set to None for an untargeted attack. 
target = None

#Norm value used to calculate reward. See np.linalg.norm(). For best results, do not set this to np.inf. 
norm = 2

#Name of the file to save attack model. Should be a .zip file. Set this to an existing filepath to continue training an old model. Set to None to not save the model.
model_name = "Models/PPO-L2.zip"

#RL Framework to train the model. Currently supports "PPO" and "TD3"
framework = "PPO"

#Frequency at which to save the model.
save_interval = 1000

#Parameter file that stores hyperparameters. Obtained from Optuna.py.
param_file = "Params/PPO-L2.pkl"

gradientRun(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, save_interval, param_file)