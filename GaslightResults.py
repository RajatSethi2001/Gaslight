import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms

from Classifiers.TorchCIFAR10 import Net
from stable_baselines3 import PPO, TD3

def distance(x_adv, x, norm=2):
    diff = x - x_adv
    diff_flat = diff.flatten()  
    return np.linalg.norm(diff_flat, norm)

def gaslight_test_mnist(attacker_path, classifier_path, target, framework="PPO", max_queries=10):
    attacker = PPO.load(attacker_path)
    classifier = tf.keras.models.load_model(classifier_path)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_test = [np.random.uniform(low=0, high=1, size=(28, 28, 1)) for _ in range(10000)]
    x_test = [x.reshape((28, 28, 1)) / 255.0 for x in x_test]

    l2_avg = 0
    linf_avg = 0
    query_avg = 0
    successes = 0
    for idx in range(len(x_test)):
        if idx % 1000 == 0:
            print(idx)
        
        # x = x_test[idx].reshape((28, 28, 1)) / 255.0
        adv = np.copy(x_test[idx])

        queries = 0
        for _ in range(max_queries):
            queries += 1
            action, _ = attacker.predict(adv)
            adv = np.clip(adv + action, 0, 1)
            adv_input = adv.reshape((1, 28, 28, 1))
            label = np.argmax(list(classifier(adv_input).numpy()[0]))

            if (target is None and label != y_test[idx]) or (target is not None and label == target):
                successes += 1
                query_avg += queries
                l2_avg += distance(adv, x_test[idx], 2)
                linf_avg += distance(adv, x_test[idx], np.inf)
                break
    
    print(f"Success Rate: {successes * 100 / len(x_test)}")
    print(f"Query Average: {query_avg / successes}")
    print(f"L2 Average: {l2_avg / successes}")
    print(f"LInf Average: {linf_avg / successes}")

def cifar10_pytorch(attacker_path, classifier_path, target, framework="PPO", max_queries=10):
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
    snapped = False
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
                    if not snapped:
                        cv2.imwrite("real.png", x * 255)
                        cv2.imwrite("fake.png", x_adv * 255)
                        snapped = True
                    successes += 1
                    query_avg += query
                    l2_avg += distance(x_adv, x, 2)
                    linf_avg += distance(x_adv, x, np.inf)
                    break
    
    print(f"Success Rate: {successes * 100 / len(testset)}")
    print(f"Query Average: {query_avg / successes}")
    print(f"L2 Average: {l2_avg / successes}")
    print(f"LInf Average: {linf_avg / successes}")

# gaslight_test_mnist("Models/PPO-L2.zip", "Classifiers/mnist", None, max_queries=1)
cifar10_pytorch("Models/CIFAR10-TD3-TargetedFull.zip", "Classifiers/cifar_net.pth", 6, "TD3", 1)
