import gym
import math
import numpy as np

from gym.spaces import Box, Tuple
from utils import similarity

class GradientEnv(gym.Env):
    def __init__(self, predict, extra, input_shape, input_range, eps, target):
        self.predict = predict
        self.extra = extra
        self.input_shape = input_shape
        self.input_range = input_range
        self.distance = input_range[1] - input_range[0]
        self.target = target

        self.eps = eps

        self.observation_space = Box(low=input_range[0], high=input_range[1], shape=input_shape, dtype=np.float32)

        self.action_space = Box(low=-self.eps, high=self.eps, shape=input_shape, dtype=np.float32)

        self.input = np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape)
        self.original = np.copy(self.input)
        self.true = self.predict(self.input, self.extra)

        self.actions = 0

    def step(self, action):
        self.actions += 1

        self.input = np.clip(self.input + action, self.input_range[0], self.input_range[1])
        label = self.predict(self.input, self.extra)

        similarity = self.input_range[1] - self.input_range[0] - np.average(abs(self.original - self.input))

        reward = 0
        if (self.target is None and label != self.true) or (self.target is not None and label == self.target):
            reward = similarity
        
        return self.input, reward, True, {}

    def reset(self):
        self.input = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.input_shape)
        self.original = np.copy(self.input)
        self.true = self.predict(self.input, self.extra)

        self.actions = 0

        return self.original

    def render(self):
        print(f"Actions: {self.actions}")