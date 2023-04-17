import torch
import torchvision.transforms as transforms

from GaslightEngine import gaslightRun
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

def predict(input_array, extra=None):
    victim = extra["model"]
    transform = extra["transform"]
    
    with torch.no_grad():
        x = transform(input_array)
        x = torch.unsqueeze(x, 0)
        outputs = victim(x.float())
        _, predicted = torch.max(outputs.data, 1)
        return int(predicted[0])

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

extra = {"model": model, "transform": transform}

#Shape of the classifier input.
input_shape = (224, 224, 3)

#Range of the inputs values. First value is min, second value is max. Preferably make this (0, 1) and scale in predict.
input_range = (0, 1)

#Maximum deviation per input parameter in a single action.
max_delta = 0.25

#Target label for attacking model to achieve. Corresponds with output from predict(). Set to None for an untargeted attack. 
target = None

#Norm value used to calculate reward. See np.linalg.norm(). For best results, do not set this to np.inf. 
norm = 2

#Name of the file to save attack agent. Should be a .zip file. Set this to an existing filepath to continue training an old model. Set to None to not save the model.
model_name = "Agents/ImageNet-PPO-Untargeted2.zip"

#RL Framework to train the model. Currently supports "PPO" and "TD3"
framework = "PPO"

#Frequency at which to save the model.
save_interval = 1000

#Parameter file that stores hyperparameters. Obtained from Optuna.py.
param_file = "Params/PPO.pkl"

gaslightRun(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, save_interval, param_file)