import json 
import numpy as np

class Pricing_Aggregate:
    def __init__(self, id):
        self.id = id
        with open('Pricing/configs/pricing_env.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]
    
        # Environment 
        self.proba = campaign["probabilities"] #non_aggregate proba
        self.prices = campaign["prices"] 

        #aggregate proba 
        len1=len(self.proba) #number of category
        len2=len(self.proba[0]) #number of proba per category, so the number of arm
        self.agg_proba=[]#AGGREGATE PROBABILITIES
        #let's do the average of the proba of the categories
        for i in range(0,len2):
            tmp=[]
            for j in range(0,len1):
                tmp.append(self.proba[j][i]) 
            self.agg_proba.append(np.mean(tmp)) 

        #aggregate demand, given a price, so x is the price 
        self.demand_functions = (lambda x, p=self.agg_proba: self.demand(x, p))

    def demand(self, x, p):
        price_to_index = {5:0, 7.5:1, 10:2, 12.5:3, 15:4, 17.5:5, 20:6, 22.5:7, 25:8, 27.5:9}
        k = price_to_index[x]
        return p[k]


def test():
    pricingenv=Pricing_Aggregate(0)
    print(pricingenv.prices)
    print(pricingenv.agg_proba)
    print(pricingenv.demand_functions(7.5))


#test()