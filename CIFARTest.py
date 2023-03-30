import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Classifiers.densenet import DenseNet
from collections import OrderedDict

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

test_data = datasets.CIFAR10(root="./data", train=False, transform=transform)
loader = torch.utils.data.DataLoader(test_data, batch_size=1)

model = DenseNet(32, (6, 12, 32, 32), 64)
state_dict = torch.load("Classifiers/densenet169.pt", map_location="cpu")
model.load_state_dict(state_dict)

correct = 0
with torch.no_grad():
    for data in loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(int(predicted[0]))
        input()
        correct += (predicted == labels).sum().item()

print(correct * 100 / len(test_data))