import json
import numpy as np
from scipy.stats import poisson
from Pricing.environment.Pricing_Config_Manager import *


class Advertising_Config_Manager:
    def __init__(self, id):
        self.id = id
        with open('Advertising/configs/sub_camp_config.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]

        # Class settings
        self.feature_labels = list(campaign["subcampaigns"].keys())

        # Experiment settings
        self.sigma = campaign["sigma"]
        self.click_functions = {}
        self.cost_functions = {}
        self.future_visits = {}
        self.future_visits2 = {}

        for feature in campaign["subcampaigns"]:
            max_click_value = campaign["subcampaigns"][feature]["max_click_value"]
            max_cost_value = campaign["subcampaigns"][feature]["max_cost_value"]
            mean_value = campaign["subcampaigns"][feature]["mean_value"]
            self.click_functions[feature] = (lambda x, m=max_click_value: self.function(x, m) )
            self.cost_functions[feature] = (lambda x, m=max_cost_value: self.cost(x, m) )
            self.future_visits[feature] = (lambda t, mean=mean_value: self.f(t, mean) )
            #self.demand_functions[feature] = (lambda x, self.demand(x , category))
            self.future_visits2[feature] = (lambda t, mean=mean_value: self.f2(t, mean) )


    #number of click function
    def function(self, x, m):
        return m * (1.0 - np.exp(-4*x+3*x**3))
    
    #cost per click
    def cost(self, x, m):
        return np.log(1+x)

    #futur visit
    def f(self, t, mean):
        return poisson.pmf(mu=mean,k=t)
    
    def f2(self,t, mean):
        return np.random.poisson(lam=mean)

    
   
        

 
