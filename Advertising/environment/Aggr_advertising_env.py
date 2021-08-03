import json
import numpy as np
from numpy.core.fromnumeric import mean
from scipy.stats import poisson



class Advertising_Aggregate:
    def __init__(self, id):
        self.id = id
        with open('Advertising/configs/sub_camp_config.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]


        # Experiment settings
        self.sigma = campaign["sigma"]
        self.click_functions = {}
        self.cost_functions = {}
        self.future_visits = {}

        #Let's aggregate : make the average of the value in sub_camp_config.json
        max_click_value=[]
        max_cost_value=[]
        mean_value=[]
        for feature in campaign["subcampaigns"]:
            max_click_value.append(campaign["subcampaigns"][feature]["max_click_value"]) #non aggregate
            max_cost_value.append(campaign["subcampaigns"][feature]["max_cost_value"])  #non aggregate
            mean_value.append(campaign["subcampaigns"][feature]["mean_value"])  #non aggregate
        max_click_value=np.mean(max_click_value)#aggregate
        max_cost_value=np.mean(max_cost_value)  #aggregate
        mean_value=np.mean(mean_value)          #aggregate
        
        #number of daily click of new new user (never click before), depending on the bid, so x is the bid 
        self.click_functions = (lambda x, m=max_click_value: self.function(x, m) ) #aggregate   
        
        #the cost given a price, so x is the price
        self.cost_functions = (lambda x, m=max_cost_value: self.cost(x, m) )       #aggregate
        
        #distribution probability the user will come back to buy, after a first purchase, t is the number of previous visit? not sure, try to confirm 
        self.future_visits = (lambda t, mean=mean_value: self.f(t, mean) )         #aggregate
        
        #other way to do an Aggregate part
        # self.click_functions_agg = np.empty((0, 10), int)
        # self.cost_functions_agg = np.empty((0, 10), int)
        # self.future_visits_agg = np.empty((0, 10), int)

        # for feature in campaign["subcampaigns"]:
        #     self.click_functions_agg = np.append(self.click_functions_agg,np.array([self.click_functions[feature]()]), axis=0)
        # np.mean(click_agg, axis=0)

    #number of click function
    def function(self, x, m):
        return m * (1.0 - np.exp(-4*x+3*x**3))


    def cost(self, x, m):
        return np.log(1+x)


    def f(self, t, mean):
        return poisson.pmf(mu=mean,k=t)

def test():
    advenv=Advertising_Aggregate(0)
    print(advenv.click_functions(0.6))
    print(advenv.cost_functions(22,5))
    print(advenv.future_visits(4))


#test()
   
        

 
