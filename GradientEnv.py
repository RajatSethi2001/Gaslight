import gym
import math
import numpy as np

from gym.spaces import Box, Tuple
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

        self.observation_space = Box(low=input_range[0], high=input_range[1], shape=input_shape, dtype=np.float32)
        self.action_space = Box(low=-self.max_delta, high=self.max_delta, shape=input_shape, dtype=np.float32)

        self.original = np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape)
        self.true_label = self.predict(self.original, self.extra)

    def step(self, action):
        adv = np.clip(self.original + action, self.input_range[0], self.input_range[1])
        label = self.predict(adv, self.extra)

        reward = 0
        if (self.target is None and label != self.true_label) or (self.target is not None and label == self.target):
            reward = distance(adv, self.original, self.norm)
        
        return adv, reward, True, {}

    def reset(self):
        self.original = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.input_shape)
        self.true_label = self.predict(self.original, self.extra)

        return self.original

    def render(self):
        pass