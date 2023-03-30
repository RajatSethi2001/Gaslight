import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms

from Classifiers.TorchCIFAR10 import Net
from ParamFinder import ParamFinder
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# #input_array - Array of values to be classified. Will have a shape of "input_shape" and range of "input_range"
# #extra - Additional argument used to pass in any other values, like the classifier model.
# #This function should return the top-1 output of the classifier. It can be any type, as long as "target" is a possible output and the type has a valid "==" operation.
# def predict(input_array, extra=None):
#     victim = extra["model"]
#     input_reshaped = input_array.reshape((1,) + input_array.shape)
#     label = np.argmax(list(victim(input_reshaped).numpy()[0]))
#     return label

# #Extra input for the predict function. Stores the classifier model so it doesn't have to be reloaded each iteration.
# extra = {"model": tf.keras.models.load_model('Classifiers/cifar10')}

def predict(input_array, extra=None):
    victim = extra["model"]
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    with torch.no_grad():
        x = transform(input_array * 255.0)
        x = torch.unsqueeze(x, 0)
        outputs = victim(x.float())
        _, predicted = torch.max(outputs.data, 1)
        return int(predicted[0])

model = Net()
state_dict = torch.load("./Classifiers/cifar_net.pth")
model.load_state_dict(state_dict)
model.eval()

extra = {"model": model}

# def predict(input_array, extra=None):
#     victim = extra["victim"]
#     preprocess = extra["preprocess"]
#     pil_image = Image.fromarray(np.uint8(input_array * 255.0))
#     input_tensor = preprocess(pil_image).unsqueeze(0)
#     outputs = victim(input_tensor.float())
#     _, predicted = torch.max(outputs.data, 1)
#     return int(predicted[0])

# weights = ResNet18_Weights.DEFAULT
# model = resnet18(weights=weights)
# model.eval()

# preprocess = weights.transforms()

# extra = {"victim": model, "preprocess": preprocess}

#Shape of the classifier input.
input_shape = (32, 32, 3)

#Range of the inputs values. First value is min, second value is max. Preferably make this (0, 1) and scale in predict.
input_range = (0, 1)

#Maximum deviation per input parameter in a single action.
max_delta = 0.3

#Target label for attacking model to achieve. Corresponds with output from predict(). Set to None for an untargeted attack. 
target = 0

#Norm value used to calculate reward. See np.linalg.norm(). For best results, do not set this to np.inf. 
norm = 2

#Find optimal hyperparameters for an already trained model to improve future iterations. If you don't have an existing model, set this to None (recommended).
model_name = None

#RL Framework to train the model. Currently supports "PPO" and "TD3"
framework = "TD3"

#Parameter file that stores hyperparameters. Can also put an existing filepath to continue trials. Should be a .pkl file.
param_file = "Params/CIFAR10-TD3-New.pkl"

#How many trials to run for this process. Each trial represents a new set of hyperparameters.
trials = 20

#How many datapoints to collect per trial. Each datapoint yields metrics that represent the success of the attack.
samples = 8

#How many timesteps to run through per sample.
timesteps = 1024

if __name__=='__main__':
    param_finder = ParamFinder(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, param_file, trials, samples, timesteps)
    param_finder.run()


