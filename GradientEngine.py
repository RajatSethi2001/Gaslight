import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from os.path import exists
from GradientEnv import GradientEnv
from stable_baselines3 import PPO, TD3, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from utils import similarity

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
    
def gradientRun(predict, extra, input_shape, input_range, eps, target, model_name, framework, param_file=None, save_interval=100):
    if framework == "PPO":
        #Hyperparameters collected from Optuna.py
        hyperparams = {}
        if param_file is not None:
            study = pickle.load(open(param_file, 'rb'))
            hyperparams = study.best_params

            if hyperparams['batch_size'] > hyperparams['n_steps']:
                hyperparams['batch_size'] = hyperparams['n_steps']
        
        # env = GradientEnv(predict, extra, input_shape, input_range, target)
        env_kwargs = {
            "predict": predict,
            "extra": extra,
            "input_shape": input_shape,
            "input_range": input_range,
            "eps": eps,
            "target": target
        }
        vec_env = make_vec_env(GradientEnv, 4, env_kwargs=env_kwargs)
        checkpoint_callback = GaslightCheckpoint(save_interval, model_name)

        if model_name is not None and exists(model_name):
            model_attack = eval(f"PPO.load(\"{model_name}\", env=vec_env, **hyperparams)")
        #RL models to use for testing.
        else:
            policy_name = "MlpPolicy"
            model_attack = PPO(policy_name, vec_env, **hyperparams)

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

        env = GradientEnv(predict, extra, input_shape, input_range, eps, target)
        checkpoint_callback = GaslightCheckpoint(save_interval, model_name)

        if model_name is not None and exists(model_name):
            model_attack = eval(f"TD3.load(\"{model_name}\", env=env, **hyperparams)")
        #RL models to use for testing.
        else:
            policy_name = "MlpPolicy"
            model_attack = TD3(policy_name, env, **hyperparams)
    
    else:
        print(f"Framework {framework} does not exist. Available frameworks are (PPO, TD3)")
        exit()
    
    originals = [np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape) for _ in range(100)]
    true_labels = [predict(x, extra) for x in originals]
    similar_list = []
    reward_list = []
    success_list = []
    timesteps = []
    plt.ion()
    figure, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(timesteps, similar_list)
    ax[0].set_title("Similarity")

    ax[1].plot(timesteps, reward_list)
    ax[1].set_title("Rewards")

    ax[2].plot(timesteps, success_list)
    ax[2].set_title("Successes")
    for _ in range(300):
        model_attack.learn(3000, progress_bar=True, callback=checkpoint_callback)
        copies = [np.copy(x) for x in originals]
        similar_avg = 0
        reward_avg = 0
        success_count = 0
        for idx in range(len(copies)):
            action, _ = model_attack.predict(copies[idx])
            copies[idx] = np.clip(copies[idx] + action, input_range[0], input_range[1])
            new_label = predict(copies[idx], extra)
            reward = input_range[1] - input_range[0] - np.average(abs(originals[idx] - copies[idx]))
            if (target is None and new_label != true_labels[idx]) or (target is not None and new_label == target):
                success_count += 1
                reward_avg += reward
                
            similar_avg += reward
            
        similar_list.append(similar_avg / len(copies))
        reward_list.append(reward_avg / len(copies))
        success_list.append(success_count)
        timesteps = list(range(len(similar_list)))

        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].plot(timesteps, similar_list)
        ax[0].set_title("Similarity")

        ax[1].plot(timesteps, reward_list)
        ax[1].set_title("Rewards")

        ax[2].plot(timesteps, success_list)
        ax[2].set_title("Successes")

        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)

    plt.show()          
