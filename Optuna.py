import torch
import torch.nn as nn
import torchvision.transforms as transforms

from Classifiers.TorchCIFAR10 import Net
from ParamFinder import ParamFinder
from PIL import Image
from torchvision.models import efficientnet_v2_s

def predict(input_array, extra=None):
    victim = extra["model"]
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    with torch.no_grad():
        x = transform(input_array)
        x = torch.unsqueeze(x, 0)
        outputs = victim(x.float())
        _, predicted = torch.max(outputs.data, 1)
        return int(predicted[0])

model = Net()
state_dict = torch.load("./Classifiers/cifar10.pth")
model.load_state_dict(state_dict)
model.eval()

extra = {"model": model}

#Shape of the classifier input.
input_shape = (32, 32, 3)

#Range of the inputs values. First value is min, second value is max. Preferably make this (0, 1) and scale in predict.
input_range = (0, 1)

#Maximum deviation per input parameter in a single action.
max_delta = 0.16

#Target label for attacking model to achieve. Corresponds with output from predict(). Set to None for an untargeted attack. 
target = 0

#Norm value used to calculate reward. See np.linalg.norm(). For best results, do not set this to np.inf. 
norm = 2

#Find optimal hyperparameters for an already trained model to improve future iterations. If you don't have an existing model, set this to None (recommended).
model_name = None

#RL Framework to train the model. Currently supports "PPO" and "TD3"
framework = "PPO"

#Parameter file that stores hyperparameters. Can also put an existing filepath to continue trials. Should be a .pkl file.
param_file = "Params/PPO-Targeted.pkl"

#How many trials to run for this process. Each trial represents a new set of hyperparameters.
trials = 20

#How many datapoints to collect per trial. Each datapoint yields metrics that represent the success of the attack.
samples = 16

#How many timesteps to run through per sample.
timesteps = 1024

if __name__=='__main__':
    param_finder = ParamFinder(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, param_file, trials, samples, timesteps)
    param_finder.run()


