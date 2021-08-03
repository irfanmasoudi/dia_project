from Pricing.environment.PricingEnvironment import Personalized_Environment
import numpy as np
import matplotlib.pyplot as plt
from Advertising.environment.Aggr_advertising_env import *
from Pricing.environment.Aggr_pricing_env import *
from Learner.TS_Learner2 import TS_Learner
from Learner.UCB import UCB
from Learner.Clairvoyant import Clairvoyant

class Experiment_3:
    def __init__(self, n_arms=10, price_env_id=0, adv_env_id=0):
         
        self.n_arms = n_arms
        self.n_rounds = 365
        self.bid=0.6 #the bid is fixed

        self.n_visits = np.arange(1,10)
        
        pricing_env = Pricing_Aggregate(price_env_id)
        advertising_env = Advertising_Aggregate(adv_env_id)


        # list of prices
        self.prices = pricing_env.prices
        self.probabilities = pricing_env.agg_proba
        #creation of personalized environment
        self.pricing_env_bis = Personalized_Environment(self.prices, self.probabilities)

        # Class settings
        #self.feature_labels = advertising_env.feature_labels

        # Click functions
        self.click_functions = advertising_env.click_functions

        # Cost functions
        self.cost_functions = advertising_env.cost_functions

        # Conversion rate functions
        self.demand_functions = pricing_env.demand_functions

        # Probabilities of future visits
        self.future_visits = advertising_env.future_visits


 
    def reward(self, price, bid):
        ѵ = np.sum(self.n_visits * self.future_visits(self.n_visits))
        v = (price * self.demand_functions(price)) * (1 + ѵ) - self.cost_functions(bid)     
        n = self.click_functions(bid) 
        return v * n

    def opt(self,bid,prices):
        table=[]
        #let's get all the reward value for this fixe bid
        for k in range(len(prices)):
            temp=self.reward(prices[k],bid) #the bid is fix
            table.append(temp)

        max_value = np.max(table)
        opt_index = np.where(table == max_value)
        return max_value
    
    def run(self) :
        opt = self.opt(self.bid, self.prices)

        T = self.n_rounds
        n_experiments = 100
        ts_rewards_per_experiment = []
        ucb_rewards_per_experiment = []
        cl_rewards_per_experiment = []

        for e in range(0, n_experiments):
            ts_learner = TS_Learner(n_arms=self.n_arms)
            ucb_learner = UCB(n_arms=self.n_arms)
            cl_learner= Clairvoyant(n_arms=self.n_arms)
            for t in range(0, T):
                # Thompson Sampling Learner
                pulled_arm = ts_learner.pull_arm()
                reward = self.reward(self.prices[pulled_arm],self.bid) 
                ts_learner.update_bis(pulled_arm, reward, opt)

                # UCB
                pulled_arm = ucb_learner.pull_arm()
                reward = self.reward(self.prices[pulled_arm],self.bid) 
                ucb_learner.update(pulled_arm, reward)

                # Clairvoyant
                pulled_arm = cl_learner.pull_arm()
                reward = self.reward(self.prices[pulled_arm],self.bid)
                cl_learner.update(pulled_arm, reward)
                
            ts_rewards_per_experiment.append(ts_learner.collected_rewards)
            ucb_rewards_per_experiment.append(ucb_learner.collected_rewards)
            cl_rewards_per_experiment.append(cl_learner.collected_rewards)
            
            
        #print(cl_rewards_per_experiment)
        plt.figure(0)
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
        plt.plot(np.cumsum(np.mean(opt - ucb_rewards_per_experiment, axis=0)), 'g')
        plt.plot(np.cumsum(np.mean(opt - cl_rewards_per_experiment, axis=0)), 'b')
        plt.legend(["TS", "UCB", "Cl"])
        plt.show()
        
        return cl_rewards_per_experiment
        

def test():
    pricing_env = Pricing_Aggregate(0)
    exp3 = Experiment_3()
    prices = pricing_env.prices
    tmp=exp3.click_functions(exp3.bid)
    print(tmp)
    
    
#test()