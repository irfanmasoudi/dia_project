from .GPTS_Learner import *
import numpy as np


class Subcampaign_Learner(GPTS_Learner):
    def __init__(self, n_arms, arms, label):
        super().__init__(n_arms, arms)
        self.label = label

    def pull_arms(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        sampled_values = np.maximum(0, sampled_values)  # avoid negative values
        return sampled_values