import numpy as np
import optuna
import pickle
import torch.nn as nn

from GradientEnv import GradientEnv
from os.path import exists
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from utils import distance

class ParamFinder:
    def __init__(self, predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework, param_file, trials, samples, timesteps):
        self.predict = predict
        self.extra = extra
        self.input_shape = input_shape
        self.input_range = input_range
        self.max_delta = max_delta
        self.target = target
        self.norm = norm
        self.model_name = model_name
        self.framework = framework
        self.param_file = param_file
        self.trials = trials
        self.samples = samples
        self.timesteps = timesteps

        #Retrieve existing parameters if they exist. If not, create a new parameter file.
        if exists(param_file):
            self.study = pickle.load(open(self.param_file, 'rb'))
        else:
            self.study = optuna.create_study(direction="maximize")

    #Main method for Optuna.
    def run(self):
        self.study.optimize(self.optimize_framework, n_trials=self.trials)
        pickle.dump(self.study, open(self.param_file, 'wb'))
    
    def get_ppo(self, trial):
        #Possible hyperparameters for the PPO framework, as determined by RL-Zoo.
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float("vf_coef", 0, 1)

        if batch_size > n_steps:
            batch_size = n_steps
        
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
        }

    def get_td3(self, trial):
        #Possible hyperparameters for the TD3 framework, as determined by RL-Zoo.
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64, 128])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5)])
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

        train_freq = trial.suggest_categorical("train_freq", [4, 8, 16, 32, 64, 128])
        gradient_steps = train_freq

        noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal"])
        noise_std = trial.suggest_float("noise_std", 0, 1)

        hyperparams = {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "tau": tau,
        }

        print(f"Current Hyperparams: {hyperparams}")

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

        #Create a new environment for the trial.
        env_kwargs = {
            "predict": self.predict,
            "extra": self.extra,
            "input_shape": self.input_shape,
            "input_range": self.input_range,
            "max_delta": self.max_delta,
            "target": self.target,
            "norm": self.norm
        }
        vec_env = make_vec_env(GradientEnv, 4, env_kwargs=env_kwargs)
        if self.framework == "PPO":
            hyperparams = {}
            net_arch = dict(pi=[256, 256], vf=[256, 256])
            hyperparams['policy_kwargs'] = dict(net_arch=net_arch)

            #Guess the optimal hyperparameters for testing in this trial.
            hyperparams = self.get_ppo(trial)

            #Make a temporary model for parameter tuning (or use an existing model).
            model = PPO("MlpPolicy", vec_env, **hyperparams)
            if self.model_name is not None and exists(self.model_name):
                model.set_parameters(self.model_name)
        
        elif self.framework == "TD3":
            hyperparams = {}
            hyperparams['policy_kwargs'] = dict(net_arch=[256, 256])

            #Guess the optimal hyperparameters for testing in this trial.
            hyperparams = self.get_td3(trial)

            #Make a temporary model for parameter tuning (or use an existing model).
            model = TD3("MlpPolicy", vec_env, **hyperparams)
            if self.model_name is not None and exists(self.model_name):
                model.set_parameters(self.model_name)
        
        else:
            print(f"Framework {self.framework} does not exist. Available frameworks are (PPO, TD3)")
            exit()

        #To measure the effectiveness of a trial, generate random test inputs to use for metric calculations.
        originals = [np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.input_shape) for _ in range(1000)]
        #Gather the "true" labels for the testing data, used for untargeted attacks.
        true_labels = [self.predict(x, self.extra) for x in originals]

        #Calculate maximum possible distortion. This helps calculate the reward such that less distortion yields higher rewards.
        self.max_reward = distance(np.ones(self.input_shape) * self.max_delta, np.zeros(self.input_shape), self.norm)
        
        #Keep a running track of rewards per sample.
        rewards = []

        #Run each sample, which trains the model for a certain amount then evaluates rewards.
        for _ in range(self.samples):
            #Run the trial for the designated number of timesteps.
            model.learn(self.timesteps, progress_bar=True)

            #Calculate the average reward for each testing input.
            reward_avg = 0
            for idx in range(len(originals)):
                #Estimate the optimal distortion/action based on the input.
                action, _ = model.predict(originals[idx])

                #Given an distortion, add it to the input and clip the parameters.
                adv = np.clip(originals[idx] + action, self.input_range[0], self.input_range[1])

                #Determine the label of the perturbed input.
                new_label = self.predict(adv, self.extra)
                
                #If the perturbation yields the intended target label (or a different label for untargeted attacks).
                if (self.target is None and new_label != true_labels[idx]) or (self.target is not None and new_label == self.target):
                    #Calculate an aggregate score for the distortion, then set the reward to a value that is inversely proportional to the distortion.
                    reward_avg += self.max_reward - distance(adv, originals[idx], self.norm)

            #Add the average reward to the running list of metrics.
            rewards.append(reward_avg)

        #The "score" for the trial is the slope of best fit, where y-axis represents rewards and the x-axis represents samples.
        x = list(range(self.samples))
        slope, _ = np.polyfit(x, rewards, 1)
        return slope