import math
import numpy as np

def distance(x_adv, x, norm=2):
    diff = x - x_adv
    diff_flat = diff.flatten()  
    return np.linalg.norm(diff_flat, norm) 

