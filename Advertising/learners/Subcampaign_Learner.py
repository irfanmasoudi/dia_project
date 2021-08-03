from .GPTS_Learner import *
import numpy as np


class Subcampaign_Learner(GPTS_Learner):
    def __init__(self, n_arms, arms, length_scale_bounds, label, alpha):
        super().__init__(n_arms, arms, length_scale_bounds, alpha)
        self.label = label
