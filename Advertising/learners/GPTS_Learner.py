from .Learner import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms, length_scale_bounds=(1e-3, 1e3), alpha=10.0):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 10
        self.pulled_arms = []
        self.alpha = alpha
        kernel = C(1.0, length_scale_bounds) * RBF(1.0, length_scale_bounds)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=False, n_restarts_optimizer=10)

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)

        x_pred = np.atleast_2d(self.arms).T
        self.means, self.sigmas = self.gp.predict(x_pred, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        idx = np.argmax(np.random.normal(self.means, self.sigmas))
        return idx

    # it's just updating the model parameters for the initial 10 sample
    # from each arm
    def learn_kernel_hyperparameters(self, samples):
        [self.update(pulled_arm, reward) for (pulled_arm, reward) in zip(np.arange(10), samples)]
