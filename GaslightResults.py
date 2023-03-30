import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from stable_baselines3 import PPO, TD3

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def distance(x_adv, x, norm=2):
    diff = x - x_adv
    diff_flat = diff.flatten()  
    return np.linalg.norm(diff_flat, norm)

def validate_cifar10_pytorch(classifier_path):
    model = Net()
    state_dict = torch.load(classifier_path)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True) 

    successes = 0
    with torch.no_grad():
        for data in testset:
            image, label = data
            image_t = torch.unsqueeze(transform(image), 0)
            outputs = model(image_t)

            _, predictions = torch.max(outputs.data, 1)

            if int(predictions[0]) == label:
                successes += 1

    print(successes * 100 / len(testset))

def gaslight_cifar10_pytorch(attacker_path, classifier_path, target, framework="PPO", max_queries=10):
    attacker = eval(f"{framework}.load(\"{attacker_path}\")")

    model = Net()
    state_dict = torch.load(classifier_path)
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True)
    
    successes = 0
    l2_avg = 0
    linf_avg = 0
    query_avg = 0
    with torch.no_grad():
        for data in testset:
            image, label = data
            x = np.array(image) / 255.0
            x_adv = np.copy(x)

            for query in range(1, max_queries+1):
                action, _ = attacker.predict(x_adv)
                x_adv = np.clip(x_adv + action, 0, 1)
                x_input = torch.unsqueeze(transform(x_adv * 255.0), 0)
                outputs = model(x_input.float())
                _, predictions = torch.max(outputs.data, 1)
            
                if (target is None and int(predictions[0]) != label) or (target is not None and int(predictions[0]) == target):
                    successes += 1
                    query_avg += query
                    l2_avg += distance(x_adv, x, 2)
                    linf_avg += distance(x_adv, x, np.inf)
                    break
    
    print(f"Success Rate: {successes * 100 / len(testset)}")
    print(f"Query Average: {query_avg / successes}")
    print(f"L2 Average: {l2_avg / successes}")
    print(f"LInf Average: {linf_avg / successes}")

# validate_cifar10_pytorch("Classifiers/cifar_net.pth")
gaslight_cifar10_pytorch("Models/CIFAR10-Attacker-Label0.zip", "Classifiers/cifar_net.pth", 0, "TD3", 5)
