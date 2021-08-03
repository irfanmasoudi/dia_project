import json
import numpy as np

class Pricing_Config_Manager:
    """
    Manage the import of the pricing configuration .json file
    """

    def __init__(self, id):
        self.id = id
        with open('Pricing/configs/pricing_env.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]

        # Features
        self.features = campaign["features"]
        self.feature_space = self.get_feature_space()

        # Environment
        self.categories = [tuple(self.features[f][c[f]] for f in self.features) for c in campaign["categories"]]
        self.prices = campaign["prices"]
        self.probabilities = campaign["probabilities"]
        self.demand_functions ={}
        

        for index, category in enumerate(self.categories):
            probabilities = self.probabilities[index]
            self.demand_functions[category] = (lambda x, p=probabilities: self.demand(x, p))

        #maybe not at the right place

    def demand(self, x, p):
        price_to_index = {5:0, 7.5:1, 10:2, 12.5:3, 15:4, 17.5:5, 20:6, 22.5:7, 25:8, 27.5:9}
        k = price_to_index[x]
        return p[k]
        
    

    def get_feature_space(self):
        """ compute the feature space """

        def get_feature_space_rec(features, feature_list, values):
            """  recursive function """

            if len(feature_list) == 0:
                feature_space.append(tuple(values))
            else:
                f = feature_list[0]
                for v in features[f]:
                    get_feature_space_rec(features, feature_list[1:], values+[v])

        feature_space = []
        features_list = list(self.features.keys())
        get_feature_space_rec(self.features, features_list, [])

        return feature_space

    def get_indexed_categories(self):
        """ create a dictionary of indexed categories """
        return {i: c for i, c in enumerate(self.categories)}

#tests
def test():
    pricingenv=Pricing_Config_Manager(0)
    print(pricingenv.features)
    print(pricingenv.categories)
    print(pricingenv.prices)
    print(pricingenv.probabilities)
    print(pricingenv.demand_functions[('y', 'f')](7.5))
    print(pricingenv.demand_functions[('a', 'f')](7.5))
    print(pricingenv.demand_functions[('y', 'u')](7.5))

#test()