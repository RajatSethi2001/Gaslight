import gym
import math
import numpy as np

from gym.spaces import Box
from utils import distance

class GradientEnv(gym.Env):
    def __init__(self, predict, extra, input_shape, input_range, max_delta, target, norm=2):
        self.predict = predict
        self.extra = extra
        self.input_shape = input_shape
        self.input_range = input_range
        self.max_delta = max_delta
        self.target = target
        self.norm = norm
        
        #Model input is an array with a shape and range designated by the victim classifier.
        self.observation_space = Box(low=input_range[0], high=input_range[1], shape=input_shape, dtype=np.float32)

        #Model output is an array with the same shape as the input, but the range is limited by max-delta and can modify a value in either direction.
        self.action_space = Box(low=-self.max_delta, high=self.max_delta, shape=input_shape, dtype=np.float32)

        #Generate a random input to train the model.
        self.original = np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape)
        
        #Determine "true" label for training sample.
        self.true_label = self.predict(self.original, self.extra)
        
        #Calculate maximum possible distortion. This helps calculate the reward such that less distortion yields higher rewards.
        self.max_reward = distance(np.ones(self.input_shape) * self.max_delta, np.zeros(self.input_shape), self.norm)

    def step(self, action):
        #Given an distortion, add it to the input and clip the parameters.
        adv = np.clip(self.original + action, self.input_range[0], self.input_range[1])

        #Determine the label of the perturbed input.
        label = self.predict(adv, self.extra)

        #By default, the reward is 0.
        reward = 0
        #If the perturbation yields the intended target label (or a different label for untargeted attacks).  
        if (self.target is None and label != self.true_label) or (self.target is not None and label == self.target):
            #Calculate an aggregate score for the distortion, then set the reward to a value that is inversely proportional to the distortion.
            reward = self.max_reward - distance(adv, self.original, self.norm)
        
        #Return the outcome of the action. Each episode is always done after one step.
        return adv, reward, True, {}

    def reset(self):
        #At the beginning of each episode, create a new sample image and determine its class.
        self.original = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.input_shape)
        self.true_label = self.predict(self.original, self.extra)

        return self.original

    def render(self):
        pass