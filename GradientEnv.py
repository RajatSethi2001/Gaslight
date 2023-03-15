import gym
import math
import numpy as np

from gym.spaces import Box, Tuple

class GradientEnv(gym.Env):
    def __init__(self, predict, extra, input_shape, input_range, target):
        self.predict = predict
        self.extra = extra
        self.input_shape = input_shape
        self.input_range = input_range
        self.distance = input_range[1] - input_range[0]
        self.target = target

        self.eps = 1.0

        self.observation_space = Box(low=input_range[0], high=input_range[1], shape=(2,) + input_shape, dtype=np.float32)

        # mins = np.array([-self.eps] + [0 for _ in range(len(input_shape))])
        # maxs = np.array([self.eps] + [1 for _ in range(len(input_shape))])
        self.action_space = Box(low=-self.eps, high=self.eps, shape=input_shape, dtype=np.float32)

        self.input = np.random.uniform(low=input_range[0], high=input_range[1], size=input_shape)
        self.original = np.copy(self.input)
        self.true = self.predict(self.input, self.extra)

        self.current_sim = 1.0

        self.actions = 0

    def step(self, action):
        self.actions += 1
        # delta = action[0]
        # location = action[1::]
        # loc_temp = []
        # for idx in range(len(location)):
        #     loc_temp.append(int(round((self.input_shape[idx] - 1) * location[idx])))
        # location = tuple(loc_temp)
        # self.input[location] = np.clip(self.input[location] + delta, self.input_range[0], self.input_range[1])

        self.input = np.clip(self.input + action, self.input_range[0], self.input_range[1])
        label = self.predict(self.input, self.extra)

        new_sim = self.similarity(self.original, self.input)
        self.current_sim = new_sim

        reward = 0
        done = False
        if label != self.true:
            reward = new_sim
            done = True
            print(f"Original: {self.true}\nFake: {label}\nSimilarity: {round(new_sim * 100, 3)}")
            self.render()
        # elif self.actions % 100 == 0:
        else:
            done = True
            print(f"Original: {self.true}\nFake: {label}\nSimilarity: {round(new_sim * 100, 3)}")
            self.render()
        
        return (self.original, self.input), reward, done, {}

    def reset(self):
        self.input = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.input_shape)
        self.original = np.copy(self.input)
        self.true = self.predict(self.input, self.extra)

        self.current_sim = 1.0
        self.actions = 0

        return (self.original, self.input)

    def render(self):
        print(f"Actions: {self.actions}")

    def similarity(self, original, perturb):
        #Similarity is measured by the distance between the original array and the perturbed array.
        range = self.input_range[1] - self.input_range[0] 
        euclid_distance = 0
        for idx, _ in np.ndenumerate(perturb):
            # Find the difference in values, normalize the value, then square it.
            value_distance = (perturb[idx] - original[idx]) ** 2
            euclid_distance += value_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        return 1 - math.sqrt(euclid_distance / (math.prod(self.input_shape) * (range ** 2)))