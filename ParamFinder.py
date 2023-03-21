import numpy as np
import optuna
import pickle

from GradientEnv import GradientEnv
from os.path import exists
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from utils import similarity

class ParamFinder:
    def __init__(self, predict, extra, input_shape, input_range, eps, target, model_name, framework, param_file, trials, samples, timesteps, mode):
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
        self.samples = samples
        self.timesteps = timesteps
        self.mode = mode

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
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

        train_freq = trial.suggest_categorical("train_freq", [8, 16, 32, 64, 128, 256, 512])
        gradient_steps = train_freq

        noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
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
        true_labels = [self.predict(x, self.extra) for x in originals]

        similarities = []
        successes = []
        rewards = []

        for _ in range(self.samples):
            #Run the trial for the designated number of timesteps.
            model.learn(self.timesteps, progress_bar=True)
            copies = [np.copy(x) for x in originals]
            success_count = 0
            similar_avg = 0
            reward_avg = 0
            for idx in range(len(copies)):
                action, _ = model.predict(copies[idx])
                copies[idx] = np.clip(copies[idx] + action, self.input_range[0], self.input_range[1])
                new_label = self.predict(copies[idx], self.extra)
                reward = self.input_range[1] - self.input_range[0] - np.average(abs(originals[idx] - copies[idx]))
                if (self.target is None and new_label != true_labels[idx]) or (self.target is not None and new_label == self.target):
                    success_count += 1
                    reward_avg += reward

                similar_avg += reward

            similar_avg /= len(originals)
            reward_avg /= len(originals)

            similarities.append(similar_avg)
            rewards.append(reward_avg)
            successes.append(success_count)

        x = list(range(self.samples))

        if self.mode == "similarity":
            slope, _ = np.polyfit(x, similarities, 1)
            return slope
        elif self.mode == "reward":
            slope, _ = np.polyfit(x, rewards, 1)
            return slope
        elif self.mode == "success":
            slope, _ = np.polyfit(x, successes, 1)
            return slope
        else:
            print(f"Mode {self.mode} does not exist. Available modes are (similarity, success, reward)")
            exit()