import numpy as np
import tensorflow as tf

from stable_baselines3 import PPO, TD3

def distance(x_adv, x, norm=2):
    diff = x - x_adv
    diff_flat = diff.flatten()  
    return np.linalg.norm(diff_flat, norm)

def gaslight_test_mnist(attacker_path, classifier_path, target, framework="PPO", max_queries=10):
    attacker = eval(f"{framework}.load(\"{attacker_path}\")")
    classifier = tf.keras.models.load_model(classifier_path)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    l2_avg = 0
    linf_avg = 0
    query_avg = 0
    successes = 0
    for idx in range(len(x_test)):
        queries = 0
        adv = np.copy(x_test[idx]) / 255.0
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
                linf_avg += distance(adv, x_test, np.inf)
                break
    
    print(f"Success Rate: {successes * 100 / len(x_test)}")
    print(f"Query Average: {queries / successes}")
    print(f"L2 Average: {l2_avg / successes}")
    print(f"LInf Average: {linf_avg / successes}")
        

