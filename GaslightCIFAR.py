import torch
import torch.nn as nn
import torchvision.transforms as transforms

from Classifiers.TorchCIFAR10 import Net
from GaslightEngine import gaslightRun

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
max_delta = 0.10

#Target label for attacking model to achieve. Corresponds with output from predict(). Set to None for an untargeted attack. 
target = 0

#Norm value used to calculate reward. See np.linalg.norm(). For best results, do not set this to np.inf. 
norm = 2

#Name of the file to save attack agent. Should be a .zip file. Set this to an existing filepath to continue training an old model. Set to None to not save the model.
model_name = "Agents/CIFAR10-PPO-Targeted.zip"

#RL Framework to train the model. Currently supports "PPO" and "TD3"
framework = "PPO"

#Frequency at which to save the model.
save_interval = 1000

#Parameter file that stores hyperparameters. Obtained from Optuna.py.
param_file = "Params/PPO.pkl"

gaslightRun(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, save_interval, param_file)