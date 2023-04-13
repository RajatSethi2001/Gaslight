import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch.nn as nn

from os.path import exists
from GradientEnv import GradientEnv
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from utils import distance

#Callback class that saves the model after a set interval of steps.
class GaslightCheckpoint(CheckpointCallback):
    def __init__(self, save_interval, rl_model):
        super().__init__(save_interval, ".", name_prefix=rl_model)
        self.save_interval = save_interval
        self.rl_model = rl_model
    
    def _on_step(self) -> bool:
        if self.save_interval > 0 and self.n_calls % self.save_interval == 0:            
            if self.rl_model is not None:
                self.model.save(self.rl_model)
        return True
    
def gradientRun(predict, extra, input_shape, input_range, max_delta, target, norm, model_name, framework="PPO", save_interval=0, param_file=None):
    if framework == "PPO":
        hyperparams = {}
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs'] = dict(net_arch=net_arch)

        #Hyperparameters collected from Optuna.py
        if param_file is not None:
            study = pickle.load(open(param_file, 'rb'))
            hyperparams = study.best_params

            if hyperparams['batch_size'] > hyperparams['n_steps']:
                hyperparams['batch_size'] = hyperparams['n_steps']
        
        #Create vectorized environment and model-saving callback.
        env_kwargs = {
            "predict": predict,
            "extra": extra,
            "input_shape": input_shape,
            "input_range": input_range,
            "max_delta": max_delta,
            "target": target,
            "norm": norm
        }
        vec_env = make_vec_env(GradientEnv, 4, env_kwargs=env_kwargs)
        checkpoint_callback = GaslightCheckpoint(save_interval, model_name)

        #Create or load attack model.
        model_attack = PPO("MlpPolicy", vec_env, **hyperparams)
        if model_name is not None and exists(model_name):
            model_attack.set_parameters(model_name)

    elif framework == "TD3":
        hyperparams = {}
        hyperparams['policy_kwargs'] = dict(net_arch=[256, 256])
        #Hyperparameters collected from Optuna.py
        if param_file is not None:
            study = pickle.load(open(param_file, 'rb'))
            hyperparams = study.best_params

            if hyperparams['noise_type'] == 'normal':
                hyperparams['action_noise'] = NormalActionNoise(
                    mean=np.zeros(input_shape), sigma=hyperparams['noise_std'] * np.ones(input_shape)
                )
            elif hyperparams['noise_type'] == 'ornstein-uhlenbeck':
                hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(input_shape), sigma=hyperparams['noise_std'] * np.ones(input_shape)
                )
            
            del hyperparams['noise_type']
            del hyperparams['noise_std']

            hyperparams['gradient_steps'] = hyperparams['train_freq']

        #Create environment and model-saving callback.
        env_kwargs = {
            "predict": predict,
            "extra": extra,
            "input_shape": input_shape,
            "input_range": input_range,
            "max_delta": max_delta,
            "target": target,
            "norm": norm
        }
        vec_env = make_vec_env(GradientEnv, 4, env_kwargs=env_kwargs)
        checkpoint_callback = GaslightCheckpoint(save_interval, model_name)

        #Create or load attack model.
        model_attack = TD3("MlpPolicy", vec_env, **hyperparams)
        if model_name is not None and exists(model_name):
            model_attack.set_parameters(model_name)
    
    else:
        print(f"Framework {framework} does not exist. Available frameworks are (PPO, TD3)")
        exit()
    
    #Generate 1000 random inputs for testing.
    originals = [np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape) for _ in range(100)]
    
    #Determine "true" labels from testing inputs.
    true_labels = [predict(x, extra) for x in originals]
    
    #Metrics used to validate attack model. Includes L2 Norm, L-Inf Norm, and Success Rate.
    timesteps = []
    l2_list = []
    linf_list = []
    success_list = []
    
    #Create subplots to visualize metrics.
    plt.ion()
    figure, ax = plt.subplots(1, 3, figsize=(18, 6))

    #Each iteration trains the attack model for a certain amount of steps. After each iteration, recalculate the metrics. 
    for timestep in range(500):
        #Train the attack model for 1000 steps.
        model_attack.learn(1000, progress_bar=True, callback=checkpoint_callback)

        #Initialize metric averages to 0.
        l2_avg = 0
        linf_avg = 0
        success_count = 0

        #For every testing input, perturb it and calculate metrics.
        for idx in range(len(originals)):
            #Find the optimal distortion/action to modify the input values.
            action, _ = model_attack.predict(originals[idx])
            adv = np.clip(originals[idx] + action, input_range[0], input_range[1])
            
            #Feed perturbed input into victim classifier and check its label.
            new_label = predict(adv, extra)

            #Calculate distance metrics.
            l2_avg += distance(adv, originals[idx], 2)
            linf_avg += distance(adv, originals[idx], np.inf)

            #Determine if the attack is successful (Either for untargeted or targeted attacks).
            if (target is None and new_label != true_labels[idx]) or (target is not None and new_label == target):
                success_count += 1
        
        #Average findings across all tests.
        timesteps.append((timestep + 1) * 1000)
        l2_list.append(l2_avg / len(originals))
        linf_list.append(linf_avg / len(originals))
        success_list.append(success_count * 100 / len(originals))
        
        #Plot the new metrics.
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].plot(timesteps, l2_list)
        ax[0].set_title("L-2")
        ax[0].set_xlabel("Timesteps")

        ax[1].plot(timesteps, linf_list)
        ax[1].set_title("L-Inf")
        ax[1].set_xlabel("Timesteps")

        ax[2].plot(timesteps, success_list)
        ax[2].set_title("Success Rate")
        ax[2].set_xlabel("Timesteps")

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

        plt.savefig(f"Graphs/Graph.png")
       
