import numpy as np
from matplotlib import pyplot as plt

# add noise to the function f we have to estimate
def generate_observation(f, x, noise_std):
    return f(x) + np.random.normal(0,noise_std, size= f(x).shape)