import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from stable_baselines3 import PPO, TD3
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from utils import distance

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

def validate_imagenet():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testset = torchvision.datasets.ImageNet(root="./Classifiers/data", split="val")

    successes = 0
    total = 0
    with torch.no_grad():
        for image, label in testset:
            tensor_input = tr(image)
            tensor_input = torch.unsqueeze(tensor_input, 0)
            outputs = model(tensor_input)
            _, predictions = torch.max(outputs.data, 1)

            if int(predictions[0]) == label:
                successes += 1
            
            total += 1
            if total % 10 == 0:
                print(successes * 100 / total)

    print(successes * 100 / len(testset))


def validate_cifar10(classifier_path):
    # model = efficientnet_v2_s()
    # num_ftrs = model.classifier[1].in_features
    # model.classifier = nn.Linear(num_ftrs, 10)

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
    
    testset = torchvision.datasets.CIFAR10(root='./Classifiers/data', train=False,
                                            download=True)
    
    examples = {}
    
    successes = 0
    l2 = []
    linf = []
    queries = []
    valid_tests = 0
    with torch.no_grad():
        for idx in range(len(testset)):
            image, label = testset[idx]
            x = np.array(image) / 255.0
            test_output = model(torch.unsqueeze(transform(x), 0).float())
            test_label = int(torch.max(test_output.data, 1)[1][0])

            if test_label == label:
                x_adv = np.copy(x)
                valid_tests += 1
                for query in range(1, max_queries+1):
                    action, _ = attacker.predict(x_adv)
                    x_adv = np.clip(x_adv + action, 0, 1)
                    x_input = torch.unsqueeze(transform(x_adv), 0)
                    outputs = model(x_input.float())
                    _, predictions = torch.max(outputs.data, 1)
                    new_label = int(predictions[0])
                    if (target is None and new_label != label) or (target is not None and new_label == target):
                        successes += 1
                        queries.append(query)
                        l2_dist = distance(x_adv, x, 2)
                        l2.append(l2_dist)
                        linf.append(distance(x_adv, x, np.inf))

                        if label not in examples or l2_dist < examples[label][2]:
                            examples[label] = [(x * 255, label), (x_adv * 255, new_label), l2_dist]
                        break
                
                if valid_tests >= 1000:
                    break
    
    print(f"Success Rate: {successes * 100 / valid_tests}")
    print(f"Query Average: {np.mean(queries)}")
    print(f"Query Median: {np.median(queries)}")
    print(f"L2 Average: {np.mean(l2)}")
    print(f"L2 Median: {np.median(l2)}")
    print(f"LInf Average: {np.mean(linf)}")
    print(f"LInf Median: {np.median(linf)}")

    for _, v in examples.items():
        cv2.imwrite(f"Evaluation/TargetedEx/Original{v[0][1]}.png", v[0][0])
        cv2.imwrite(f"Evaluation/TargetedEx/Modified{v[0][1]}to{v[1][1]}.png", v[1][0])

def gaslight_imagenet(attacker_path, target, framework="PPO", max_queries=10):
    attacker = eval(f"{framework}.load(\"{attacker_path}\")")

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.eval()

    tr1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    tr2 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    testset = torchvision.datasets.ImageNet(root="./Classifiers/data", split="val")

    examples = {}

    successes = 0
    l2 = []
    linf = []
    queries = []
    valid_tests = 0
    with torch.no_grad():
        for idx in [random.randint(0, len(testset) - 1) for _ in range(100)]:
            image, label = testset[idx]
            x = tr1(image)
    
            test_output = model(torch.unsqueeze(tr2(x), 0).float())
            test_label = int(torch.max(test_output.data, 1)[1][0])

            if test_label == label:
                x_np = x.permute(1, 2, 0)
                x_np = np.array(x_np)
                
                x_adv = np.copy(x_np)

                valid_tests += 1
                for query in range(1, max_queries+1):
                    action, _ = attacker.predict(x_adv)
                    x_adv = np.clip(x_adv + action, 0, 1)

                    x_input = transforms.ToTensor()(x_adv)
                    x_input = torch.unsqueeze(tr2(x_input), 0)
                    outputs = model(x_input.float())
                    _, predictions = torch.max(outputs.data, 1)
                    new_label = int(predictions[0])
                    if (target is None and new_label != label) or (target is not None and new_label == target):
                        successes += 1
                        queries.append(query)
                        l2_dist = distance(x_adv, x_np, 2)
                        l2.append(l2_dist)
                        linf.append(distance(x_adv, x_np, np.inf))

                        if label not in examples or l2_dist < examples[label][2]:
                            examples[label] = [(x_np * 255, label), (x_adv * 255, new_label), l2_dist]
                        break
                
                if valid_tests >= 10:
                    break
    
    print(f"Success Rate: {successes * 100 / valid_tests}")
    print(f"Query Average: {np.mean(queries)}")
    print(f"Query Median: {np.median(queries)}")
    print(f"L2 Average: {np.mean(l2)}")
    print(f"L2 Median: {np.median(l2)}")
    print(f"LInf Average: {np.mean(linf)}")
    print(f"LInf Median: {np.median(linf)}")

    for _, v in examples.items():
        cv2.imwrite(f"Evaluation/ImageNetEx/Original{v[0][1]}.png", v[0][0])
        cv2.imwrite(f"Evaluation/ImageNetEx/Modified{v[0][1]}to{v[1][1]}.png", v[1][0])


# validate_cifar10("Classifiers/cifar10.pth")
# gaslight_cifar10_pytorch("./Agents/CIFAR10-PPO-Targeted.zip", "./Classifiers/cifar10.pth", 0, "PPO", 5)
# validate_imagenet()
gaslight_imagenet("./Agents/ImageNet-PPO-Untargeted.zip", None, "PPO", 1)