import math
import numpy as np

def similarity(original, perturb, input_range):
        #Similarity is measured by the distance between the original array and the perturbed array.
        range = input_range[1] - input_range[0] 
        euclid_distance = 0
        for idx, _ in np.ndenumerate(perturb):
            # Find the difference in values, normalize the value, then square it.
            value_distance = (perturb[idx] - original[idx]) ** 2
            euclid_distance += value_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        return 1 - math.sqrt(euclid_distance / (math.prod(original.shape) * (range ** 2)))
