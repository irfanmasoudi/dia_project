import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Advertising.environment.Advertising_Config_Manager import *
from Advertising.environment.CampaignEnvironment import *
from Advertising.environment.Advertising_Config_Manager import *
from Advertising.learners.Subcampaign_Learner import *
from experiment_1 import *


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import seaborn as sns

# class definition

class Experiment_2:
    def __init__(self, n_arms=10, price_env_id=0, adv_env_id=0):
        # env definition
        advertising_env = Advertising_Config_Manager(adv_env_id)

        # param definition
        self.n_arms = n_arms
        self.price_env_id=price_env_id
        self.adv_env_id=adv_env_id
        self.feature_labels = advertising_env.feature_labels
        self.click_sigma = 10
        self.cost_sigma = .2
        self.cost_noise_std = 0.2
        self.nclick_noise_std = 10.0

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


    def run_learner(self, gb_graphs=False):
        exp1 = Experiment_1(n_arms=self.n_arms, price_env_id=self.price_env_id,
                            adv_env_id=self.adv_env_id)

        T = 60 # period of study
        n_experiment = 10 # number of experiment

        self.gpts_click_rewards_per_experiment = [[] for i in range(3)]
        self.gpts_cost_rewards_per_experiment = [[] for i in range(3)]

        for e in range(0, n_experiment):
            # create the environment
            env = Campaign(self.bids, click_sigma=self.click_sigma, cost_sigma=self.cost_sigma)

            # list of GP-learners
            subc_click_learners = []
            subc_cost_learners = []

            for subc_id, feature_label in enumerate(self.feature_labels):
                env.add_subcampaign(label=feature_label, click_function=self.click_functions[feature_label],
                                         cost_function=self.cost_functions[feature_label])
                # Learner
                click_learner = Subcampaign_Learner(n_arms=self.n_arms, arms=self.bids, length_scale_bounds=(1e-3, 1e3),
                                                    label=feature_label, alpha=self.nclick_noise_std)
                cost_learner = Subcampaign_Learner(n_arms=self.n_arms, arms=self.bids, length_scale_bounds=(1e-3, 1e3),
                                                   label=feature_label, alpha=self.cost_noise_std)

                clicks, costs = env.subcampaigns[subc_id].round_all()
                click_learner.learn_kernel_hyperparameters(clicks)
                cost_learner.learn_kernel_hyperparameters(costs)

                subc_click_learners.append(click_learner)
                subc_cost_learners.append(cost_learner)

            for t in range(10, T):
                # sample clicks estimations from GP-learners
                # and build the Knapsack table
                click_estimate_per_subcampaign = []
                cost_estimate_per_subcampaign = []
                for subc_id, feature_label in enumerate(self.feature_labels):
                    click_estimate = subc_click_learners[subc_id].means
                    cost_estimate = subc_cost_learners[subc_id].means
                    click_estimate_per_subcampaign.append(click_estimate)
                    cost_estimate_per_subcampaign.append(cost_estimate)


                # optimizer return a list of the best bid and price for each subcampaign
                super_arms = exp1.run_with_estimates(click_estimate_per_subcampaign, cost_estimate_per_subcampaign)

                super_arm_reward = 0
                for subc_id, feature_label in enumerate(self.feature_labels):
                    # Gaussian Thompson Sampling Learner
                    # best_bid = subc_click_learners[subc_id].pull_arm()

                    # choosing the best bid for each Subcampaign
                    best_bid = super_arms[subc_id][0][0]

                    clicks, costs = env.subcampaigns[subc_id].round(best_bid)

                    subc_click_learners[subc_id].update(best_bid, clicks)
                    subc_cost_learners[subc_id].update(best_bid, costs)

            for subc_id, feature_label in enumerate(self.feature_labels):
                self.gpts_click_rewards_per_experiment[subc_id].append(subc_click_learners[subc_id].collected_rewards)
                self.gpts_cost_rewards_per_experiment[subc_id].append(subc_cost_learners[subc_id].collected_rewards)

            if gb_graphs:
                self.plot_GP_graphs(subc_click_learners, subc_cost_learners)

        return "done"

    # def run_regret(self):
    #     opt = np.max(self.env.subcampaigns[0].click_means)
    #     plt.figure(0)
    #     plt.xlabel("t")
    #     plt.ylabel("Regret")
    #     plt.plot(np.cumsum(np.mean(opt - self.gpts_click_rewards_per_experiment[0], axis=0)), 'g')
    #     plt.legend(["GPTS"])
    #     return plt.show()
        

    def plot_GP_graphs(self, subc_click_learners, subc_cost_learners):
        # plot for number of click
        for i, subc_learner in enumerate(subc_click_learners):
            x_pred = np.atleast_2d(subc_learner.arms).T
            x = np.atleast_2d(subc_learner.pulled_arms).T
            y = subc_learner.collected_rewards
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
            plt.plot(x_pred, self.cost_functions[subc_learner.label](x_pred), 'r:', label=r'$n(x)$')
            plt.plot(x.ravel(), y, 'ro', label=u'Observed Cost')
            plt.plot(x_pred, subc_learner.means, 'b-', label=u'Predicted Cost')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                    np.concatenate([subc_learner.means - 1.96*subc_learner.sigmas, (subc_learner.means + 1.96*subc_learner.sigmas)[::-1]]),
                    alpha = .5, fc='b', ec='None', label = '95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc='lower right')
            plt.show()


