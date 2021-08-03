import numpy as np


class Campaign:
    def __init__(self, bids, click_sigma=0.0, cost_sigma=0.0):
        self.subcampaigns = []
        self.bids = bids
        self.click_sigma = click_sigma
        self.cost_sigma = cost_sigma

    def add_subcampaign(self, label, click_function, cost_function):
        self.subcampaigns.append(
            Subcampaign(label, self.bids, click_function, cost_function, self.click_sigma, self.cost_sigma)
        )

    # round a specific arm
    def round(self, subcampaign_id, pulled_arm, aggregate=None):
        return self.subcampaigns[subcampaign_id].round(pulled_arm, aggregate)

    # round all subcampaigns
    def round_all(self, aggregate=None):
        table = []
        for subcampaign in self.subcampaigns:
            table.append(subcampaign.round_all(aggregate))
        return table

class Subcampaign:
    def __init__(self, label, bids, click_function, cost_function, click_sigma, cost_sigma):
        self.label = label
        self.bids = bids
        self.click_means = click_function(bids)
        self.click_sigmas = np.ones(len(bids)) * click_sigma
        self.cost_means = cost_function(bids)
        self.cost_sigmas = np.ones(len(bids)) * cost_sigma

    def round(self, pulled_arm, aggregate=None):
        # aggregate sample
        if aggregate is None:
            return np.random.normal(self.click_means[pulled_arm], self.click_sigmas[pulled_arm]), \
                   np.random.normal(self.cost_means[pulled_arm], self.cost_sigmas[pulled_arm])
        # disaggregate sample

        # round all arms

    def round_all(self, aggregate=None):
        result = [self.round(pulled_arm, aggregate) for pulled_arm in range(len(self.bids))]
        return  list(zip(*result))