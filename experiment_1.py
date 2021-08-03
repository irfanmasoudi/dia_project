from Advertising.environment.Advertising_Config_Manager import *
from Pricing.environment.Pricing_Config_Manager import *
from Advertising.optimizer.optimizer import *
import numpy as np
import matplotlib.pyplot as plt

class Experiment_1:
    def __init__(self, bids=np.linspace(0.0,1.0,10),n_arms=10, price_env_id=0, adv_env_id=0):
        self.n_arms = n_arms
        min_bid = 0.0
        max_bid = 1.0

        self.bids = bids
        self.n_visits = np.arange(1,10)

        
        pricing_env = Pricing_Config_Manager(price_env_id)
        advertising_env = Advertising_Config_Manager(adv_env_id)

        # list of prices
        self.prices = pricing_env.prices


        # Class settings
        self.feature_labels = advertising_env.feature_labels

        # Click functions
        self.click_functions = advertising_env.click_functions

        # Cost functions
        self.cost_functions = advertising_env.cost_functions

        # Conversion rate functions
        self.demand_functions = pricing_env.probabilities

        # Future visits
        self.future_visits = advertising_env.future_visits



    def run(self):
        """
        Optimization Problem Solution
        :return: list of optimal bid and price for each sub-campaign
        """
        N = len(self.feature_labels)

        table = [[] for row in range(N)]
        for j,label in enumerate(self.feature_labels):
            temp = [[0 for x in range(len(self.bids))] for y in range(len(self.prices))] 
            for k in range(len(self.prices)):
                for i in range(len(self.bids)):
                    ѵ = np.sum(self.n_visits * self.future_visits[label](self.n_visits))
                    v = (self.prices[k] * self.demand_functions[j][k]) * (1 + ѵ) - self.cost_functions[label](self.bids[i])
                    n = self.click_functions[label](self.bids[i])
                    temp[k][i] = v * n
            table[j] = temp

        opt_indexes = optimizer(table)

        return table, opt_indexes


    def run_with_estimates(self, click_estimates, cost_estimates):
        """
        Optimization Problem Solution
        :return: list of optimal bid and price for each sub-campaign
        """
        N = len(self.feature_labels)

        table = [[] for row in range(N)]
        for j,label in enumerate(self.feature_labels):
            temp = [[0 for x in range(len(self.bids))] for y in range(len(self.prices))]
            for k in range(len(self.prices)):
                for i in range(len(self.bids)):
                    ѵ = np.sum(self.n_visits * self.future_visits[label](self.n_visits))
                    v = (self.prices[k] * self.demand_functions[j][k]) * (1 + ѵ) - cost_estimates[j][i]
                    n = click_estimates[j][i]
                    temp[k][i] = v * n
            table[j] = temp

        opt_indexes = optimizer(table)

        return table, opt_indexes
    
    def run_with_estimates2(self, click_estimates, cost_estimates, convrate_estimates, ncb_estimates):
        """
        Optimization Problem Solution
        :return: list of optimal bid and price for each sub-campaign
        """
        N = len(self.feature_labels)

        table = [[] for row in range(N)]
        for j,label in enumerate(self.feature_labels):
            temp = [[0 for x in range(len(self.bids))] for y in range(len(self.prices))]
            for k in range(len(self.prices)):
                for i in range(len(self.bids)):
                    ѵ = ncb_estimates[j]
                    v = (self.prices[k] * convrate_estimates[j][k]) * (1 + ѵ) - cost_estimates[j][i]
                    n = click_estimates[j][i]
                    temp[k][i] = v * n
            table[j] = temp

        opt_indexes = optimizer(table)

        return opt_indexes

#the function test() is just for some building test
def test():
    
    exp1 = Experiment_1()
    opt, table = exp1.run_with_estimates()
    print(table)


    

#test()