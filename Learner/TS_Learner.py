from Learner.Learner import Learner
import numpy as np


class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        for i in range(n_arms):
            self.beta_parameters[i][1] = 0
        self.means = [self.beta_parameters[i,0]/(self.beta_parameters[i,0]+self.beta_parameters[i,1]) for i in range(n_arms)]

    def pull_arm(self):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (1.0 - reward)
        self.means[pulled_arm]=self.beta_parameters[pulled_arm,0]/(self.beta_parameters[pulled_arm,0]+self.beta_parameters[pulled_arm,1])

    def update_bis(self, pulled_arm, reward, best_reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (best_reward - reward)
        