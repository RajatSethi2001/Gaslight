import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

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
        #Hyperparameters collected from Optuna.py
        hyperparams = {}
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        hyperparams['policy_kwargs'] = dict(net_arch=net_arch)
        if param_file is not None:
            study = pickle.load(open(param_file, 'rb'))
            hyperparams = study.best_params

            if hyperparams['batch_size'] > hyperparams['n_steps']:
                hyperparams['batch_size'] = hyperparams['n_steps']
        
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

        model_attack = PPO("MlpPolicy", vec_env, **hyperparams)
        if model_name is not None and exists(model_name):
            model_attack.set_parameters(model_name)

    elif framework == "TD3":
        #Hyperparameters collected from Optuna.py
        hyperparams = {}
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

        env = GradientEnv(predict, extra, input_shape, input_range, max_delta, target, norm)
        checkpoint_callback = GaslightCheckpoint(save_interval, model_name)

        model_attack = TD3("MlpPolicy", env, **hyperparams)
        if model_name is not None and exists(model_name):
            model_attack.set_parameters(model_name)
    
    else:
        print(f"Framework {framework} does not exist. Available frameworks are (PPO, TD3)")
        exit()
    
    originals = [np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape) for _ in range(1000)]
    true_labels = [predict(x, extra) for x in originals]

    timesteps = []
    l2_list = []
    linf_list = []
    success_list = []
    
    plt.ion()
    figure, ax = plt.subplots(1, 3, figsize=(18, 6))
    for timestep in range(500):
        model_attack.learn(5000, progress_bar=True, callback=checkpoint_callback)
        l2_avg = 0
        linf_avg = 0
        success_count = 0
        for idx in range(len(originals)):
            action, _ = model_attack.predict(originals[idx])
            adv = np.clip(originals[idx] + action, input_range[0], input_range[1])
            new_label = predict(adv, extra)

            l2_avg += distance(adv, originals[idx], 2)
            linf_avg += distance(adv, originals[idx], np.inf)
            if (target is None and new_label != true_labels[idx]) or (target is not None and new_label == target):
                success_count += 1

        timesteps.append(timestep)
        l2_list.append(l2_avg / len(originals))
        linf_list.append(linf_avg / len(originals))
        success_list.append(success_count)
        
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].plot(timesteps, l2_list)
        ax[0].set_title("L-2")

        ax[1].plot(timesteps, linf_list)
        ax[1].set_title("L-Inf")

        ax[2].plot(timesteps, success_list)
        ax[2].set_title("Successes")

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
       
