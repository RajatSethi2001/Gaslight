import cv2
import numpy as np
import optuna
import pickle
import tensorflow as tf

from GradientEnv import GradientEnv
from os.path import exists
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from tensorflow.keras import models
from torch import nn as nn
from utils import similarity

def predict(image, extra=None):
    # tf.keras.backend.clear_session()
    # gc.collect()
    victim = extra["model"]
    image_input = image.reshape((1,) + image.shape)
    image_output = np.argmax(list(victim(image_input).numpy()[0]))
    return image_output

victim = models.load_model('Classifiers/mnist')

extra = {"model": victim}

input_shape = (28, 28, 1)

input_range = (0, 1)

eps = 0.25

target = None

model_name = None

framework = "PPO"

param_file = "Params/PPO_25.pkl"

#How many trials to run for this iteration.
trials = 20

#How many timesteps to run through per trial.
timesteps = 20000

class ParamFinder:
    def __init__(self, predict, extra, input_shape, input_range, eps, target, model_name, framework, param_file, trials, timesteps):
        self.predict = predict
        self.extra = extra
        self.input_shape = input_shape
        self.input_range = input_range
        self.eps = eps
        self.target = target
        self.model_name = model_name
        self.framework = framework
        self.param_file = param_file
        self.trials = trials
        self.timesteps = timesteps

        #Retrieve existing parameters if they exist.
        if exists(param_file):
            self.study = pickle.load(open(self.param_file, 'rb'))
        else:
            self.study = optuna.create_study(direction="maximize")

    def run(self):
        self.study.optimize(self.optimize_framework, n_trials=self.trials)
        pickle.dump(self.study, open(self.param_file, 'wb'))
    
    def get_ppo(self, trial):
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float("vf_coef", 0, 1)
        net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])

        if batch_size > n_steps:
            batch_size = n_steps

        net_arch = {
            "small": dict(pi=[64, 64], vf=[64, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256]),
        }[net_arch]
        
        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {"net_arch": net_arch}
        }

    def get_td3(self, trial):
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
        gradient_steps = train_freq

        noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
        noise_std = trial.suggest_float("noise_std", 0, 1)

        net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

        net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
        }[net_arch]

        hyperparams = {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "policy_kwargs": dict(net_arch=net_arch),
            "tau": tau,
        }

        if noise_type == "normal":
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(self.input_shape), sigma=noise_std * np.ones(self.input_shape)
            )
        elif noise_type == "ornstein-uhlenbeck":
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(self.input_shape), sigma=noise_std * np.ones(self.input_shape)
            )
        
        return hyperparams

    def optimize_framework(self, trial):
        #Save the current study in the pickle file.
        pickle.dump(self.study, open(self.param_file, 'wb'))

        env = GradientEnv(self.predict, self.extra, self.input_shape, self.input_range, self.eps, self.target)
        
        if self.framework == "PPO":
            hyperparams = {}
            #Guess the optimal hyperparameters for testing in this trial.
            hyperparams = self.get_ppo(trial)
            if self.model_name is not None and exists(self.model_name):
                model = eval(f"PPO.load(\"{self.model_name}\", env=env, **hyperparams)")
            #RL models to use for testing.
            else:
                policy_name = "MlpPolicy"
                model = PPO(policy_name, env, **hyperparams)
        
        elif self.framework == "TD3":
            hyperparams = {}
            #Guess the optimal hyperparameters for testing in this trial.
            hyperparams = self.get_td3(trial)
            if self.model_name is not None and exists(self.model_name):
                model = eval(f"TD3.load(\"{self.model_name}\", env=env, **hyperparams)")
            #RL models to use for testing.
            else:
                policy_name = "MlpPolicy"
                model = TD3(policy_name, env, **hyperparams)
        
        else:
            print(f"Framework {self.framework} does not exist. Available frameworks are (PPO, TD3)")
            exit()

        #Return the best reward as the score for this trial.
        originals = [np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.input_shape) for _ in range(100)]
        true_labels = [predict(x, self.extra) for x in originals]

        max_score = 0
        increases = 0

        samples = 20
        for _ in range(samples):
            #Run the trial for the designated number of timesteps.
            model.learn(self.timesteps / samples, progress_bar=True)
            copies = [np.copy(x) for x in originals]
            success_count = 0
            similar_avg = 0
            for idx in range(len(copies)):
                misclass = False
                action_count = 0
                while not misclass and action_count < 1:
                    action_count += 1
                    action, _ = model.predict(copies[idx])
                    copies[idx] = np.clip(copies[idx] + action, self.input_range[0], self.input_range[1])
                    new_label = self.predict(copies[idx], self.extra)

                    if (self.target is None and new_label != true_labels[idx]) or (self.target is not None and new_label == self.target):
                        misclass = True
                        success_count += 1

                similar_avg += self.input_range[1] - self.input_range[0] - np.average(abs(originals[idx] - copies[idx]))

            similar_avg /= len(originals)
            score = success_count + similar_avg

            if score > max_score:
                increases += 1
                max_score = score

        return increases

if __name__=='__main__':
    param_finder = ParamFinder(predict, extra, input_shape, input_range, eps, target, model_name, framework, param_file, trials, timesteps)
    param_finder.run()


