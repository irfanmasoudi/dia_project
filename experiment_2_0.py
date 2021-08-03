import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Advertising.environment.Advertising_Config_Manager import *
from Advertising.environment.CampaignEnvironment import *
from Advertising.environment.Advertising_Config_Manager import *
from Advertising.learners.Subcampaign_Learner import *
from experiment_1 import *
from Learner.TS_Learner import *
from Pricing.environment.PricingEnvironment import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import seaborn as sns
import random

# class definition

class Experiment_2:
    def __init__(self, n_arms=10, price_env_id=0, adv_env_id=0):
        # env definition
        advertising_env = Advertising_Config_Manager(adv_env_id)
        self.advertising_env = advertising_env
        pricing_config_manager = Pricing_Config_Manager(price_env_id)
        self.pricing_env=Personalized_Environment(np.ones(10), pricing_config_manager.probabilities)

        # param definition
        self.n_arms = n_arms
        self.price_env_id=price_env_id
        self.adv_env_id=adv_env_id
        self.feature_labels = advertising_env.feature_labels
        self.click_sigma = 10
        self.cost_sigma = .2
        self.cost_noise_std = 0.2
        self.nclick_noise_std = 10.0
        self.convrate_noise_std = 1

        # bid definition
        min_bid = 0.0
        max_bid = 1.0
        self.bids=np.linspace(min_bid, max_bid, n_arms)

        # Click functions 
        self.click_functions = advertising_env.click_functions
        # Cost functions 
        self.cost_functions = advertising_env.cost_functions

        # Rewards for each experiment (each element is a list of T rewards)
        self.gpts_click_rewards_per_experiment = [[] for i in range(3)]
        self.gpts_cost_rewards_per_experiment = [[] for i in range(3)]
        self.ts_convrate_rewards_per_experiment = [[] for i in range(3)]

    def run_learner(self, gb_graphs=False, ts_graphs=False, show_ncbs=False):
        exp1 = Experiment_1(n_arms=self.n_arms, price_env_id=self.price_env_id,
                            adv_env_id=self.adv_env_id)

        T = 365 # period of study
        n_experiment = 1 # number of experiment

        self.gpts_click_rewards_per_experiment = [[] for i in range(3)]
        self.gpts_cost_rewards_per_experiment = [[] for i in range(3)]
        self.ts_convrate_rewards_per_experiment = [[] for i in range(3)]


        for e in range(0, n_experiment):
            # create the environment
            env = Campaign(self.bids, click_sigma=self.click_sigma, cost_sigma=self.cost_sigma)

            # list of GP-learners
            subc_click_learners = []
            subc_cost_learners = []

            #list of thomson sampling learners
            subc_convrate_learners = []

            #lists for learning of Poisson law
            observations_ncb = [[0 for j in range(10)] for i in range(3)]
            means_ncb = [1 for i in range(3)]



            for subc_id, feature_label in enumerate(self.feature_labels):
                env.add_subcampaign(label=feature_label, click_function=self.click_functions[feature_label],
                                         cost_function=self.cost_functions[feature_label])
                #GPTS Learner
                click_learner = Subcampaign_Learner(n_arms=self.n_arms, arms=self.bids, length_scale_bounds=(1e-3, 1e3),
                                                    label=feature_label, alpha=self.nclick_noise_std)
                cost_learner = Subcampaign_Learner(n_arms=self.n_arms, arms=self.bids, length_scale_bounds=(1e-3, 1e3),
                                                   label=feature_label, alpha=self.cost_noise_std)
                #Thomson learner
                convrate_learner = TS_Learner(n_arms=self.n_arms)

                probabilities_future_visits = [[self for j in range(10)] for i in range(3)]

                clicks, costs = env.subcampaigns[subc_id].round_all()
                click_learner.learn_kernel_hyperparameters(clicks)
                cost_learner.learn_kernel_hyperparameters(costs)
                
                #Not sure we have to initialize it too
                #convrates = ?
                #convrate_learner.learn_kernel_hyperparameters(convrates) ?


                subc_click_learners.append(click_learner)
                subc_cost_learners.append(cost_learner)
                subc_convrate_learners.append(convrate_learner)

            for t in range(10, T):
                # sample clicks estimations from GP-learners
                # and build the Knapsack table
                click_estimate_per_subcampaign = []
                cost_estimate_per_subcampaign = []
                convrate_estimate_per_subcampaign = []
                
                for subc_id, feature_label in enumerate(self.feature_labels):
                    click_estimate = subc_click_learners[subc_id].means
                    cost_estimate = subc_cost_learners[subc_id].means

                    #moyenne de loi beta = alpha/(alpha+beta)
                    convrate_estimate = subc_convrate_learners[subc_id].means

                    click_estimate_per_subcampaign.append(click_estimate)
                    cost_estimate_per_subcampaign.append(cost_estimate)
                    convrate_estimate_per_subcampaign.append(convrate_estimate)
                    
                # optimizer return a list of the best bid and price for each subcampaign
                super_arms = exp1.run_with_estimates2(click_estimate_per_subcampaign, cost_estimate_per_subcampaign, convrate_estimate_per_subcampaign, means_ncb)

                super_arm_reward = 0
                for subc_id, feature_label in enumerate(self.feature_labels):
                    # Gaussian Thompson Sampling Learner
                    # best_bid = subc_click_learners[subc_id].pull_arm()

                    # choosing the best bid for each Subcampaign
                    best_bid = super_arms[subc_id][1][0]
                    # choosing the best price for each subcampaign
                    best_price = super_arms[subc_id][0][0]

                    clicks, costs = env.subcampaigns[subc_id].round(best_bid)

                    subc_click_learners[subc_id].update(best_bid, clicks)
                    subc_cost_learners[subc_id].update(best_bid, costs)
                    #for each click, there is a probability that the click results in a purchase
                    for _ in range(int(clicks)):
                        reward = self.pricing_env.round(subc_id, best_price)
                        subc_convrate_learners[subc_id].update(best_price, reward)
                        subc_convrate_learners[subc_id].update_observations(best_price, reward)
                        observations_ncb[subc_id].append(self.advertising_env.future_visits2[feature_label](t))
                    
                    means_ncb[subc_id]=mean(observations_ncb[subc_id])

            for subc_id, feature_label in enumerate(self.feature_labels):
                self.gpts_click_rewards_per_experiment[subc_id].append(subc_click_learners[subc_id].collected_rewards)
                self.gpts_cost_rewards_per_experiment[subc_id].append(subc_cost_learners[subc_id].collected_rewards)
                self.ts_convrate_rewards_per_experiment[subc_id].append(subc_convrate_learners[subc_id].collected_rewards)
            

            if gb_graphs:
                self.plot_GP_graphs(subc_click_learners, subc_cost_learners, self.feature_labels)

            if ts_graphs:
                self.plot_TS_graph(subc_convrate_learners)

            if show_ncbs:
                print("Mean of Future visits (lambda) for each subcampaign")
                print(means_ncb)

            

        return "done"

    # def run_regret(self):
    #     opt = np.max(self.env.subcampaigns[0].click_means)
    #     plt.figure(0)
    #     plt.xlabel("t")
    #     plt.ylabel("Regret")
    #     plt.plot(np.cumsum(np.mean(opt - self.gpts_click_rewards_per_experiment[0], axis=0)), 'g')
    #     plt.legend(["GPTS"])
    #     return plt.show()
        

    def plot_GP_graphs(self, subc_click_learners, subc_cost_learners, labels):
        # plot for number of click
        for i, subc_learner in enumerate(subc_click_learners):
            x_pred = np.atleast_2d(subc_learner.arms).T
            x = np.atleast_2d(subc_learner.pulled_arms).T
            y = subc_learner.collected_rewards
            print(labels[i])
            plt.plot(x_pred, self.click_functions[subc_learner.label](x_pred), 'r:', label=r'$n(x)$')
            plt.plot(x.ravel(), y, 'ro', label=u'Observed Clicks')
            plt.plot(x_pred, subc_learner.means, 'b-', label=u'Predicted Clicks')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                     np.concatenate([subc_learner.means - 1.96 * subc_learner.sigmas,
                                     (subc_learner.means + 1.96 * subc_learner.sigmas)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc='lower right')
            plt.show()

        #plot for cost
        for i, subc_learner in enumerate(subc_cost_learners):
            x_pred = np.atleast_2d(subc_learner.arms).T
            x = np.atleast_2d(subc_learner.pulled_arms).T
            y = subc_learner.collected_rewards
            print(labels[i])
            plt.plot(x_pred, self.cost_functions[subc_learner.label](x_pred), 'r:', label=r'$c(x)$')
            plt.plot(x.ravel(), y, 'ro', label=u'Observed Cost')
            plt.plot(x_pred, subc_learner.means, 'b-', label=u'Predicted Cost')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                    np.concatenate([subc_learner.means - 1.96*subc_learner.sigmas, (subc_learner.means + 1.96*subc_learner.sigmas)[::-1]]),
                    alpha = .5, fc='b', ec='None', label = '95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc='lower right')
            plt.show()

    # def plot_TS_graph(self, subc_learners):
    #     for i, subc_learner in enumerate(subc_learners):
    #         x_pred = np.atleast_2d(subc_learner.arms).T
    #         x = np.atleast_2d(subc_learner.pulled_arms).T
    #         y = subc_learner.collected_rewards
    #         plt.plot(x_pred, self.click_functions[subc_learner.label](x_pred), 'r:', label=r'$n(x)$')
    #         plt.plot(x.ravel(), y, 'ro', label=u'Observed conversion rate')
    #         plt.plot(x_pred, subc_learner.means, 'b-', label=u'Predicted conversion rate')

    #         plt.xlabel('$x$')
    #         plt.ylabel('$n(x)$')
    #         plt.legend(loc='lower right')
    #         plt.show()

    def plot_TS_graph(self, subc_learners):
        for i, subc_learner in enumerate(subc_learners):
            print("Learned probabilities")
            print(subc_learner.means)
            print("True Probabilities")
            print(self.pricing_env.probabilities[i])

def get_means(observations):
    means=[[0 for i in range(len(observations[0]))] for j in range(len(observations))]
    for j in range(len(observations)):
        total_n = sum(observations[j])
        for i in range(len(observations[0])):
            means[j][i]=observations[j][i]/total_n
    return means


# def choose_ncb(probabilities):
#     rand_float = random.uniform()
#     sum_probabilities = [sum(probabilities[0:i]) for i in range(len(probabilities))]
#     ncb = 0
#     while(rand_float>sum_probabilities[ncb]):
#         ncb+=1
#     return ncb

def mean(list):
    n = len(list)
    if n == 0:
        return 0
    else: return sum(list)/n
