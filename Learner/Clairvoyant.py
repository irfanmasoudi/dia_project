from Learner.Learner import Learner

class Clairvoyant(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        
    def pull_arm(self):
        return 2 # give the best reward
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)