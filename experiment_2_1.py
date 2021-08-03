from Advertising.environment.Advertising_Config_Manager import *
from Pricing.environment.Pricing_Config_Manager import *
from Advertising.optimizer.optimizer import *
import numpy as np
import matplotlib.pyplot as plt
from GaussianProcess.Gaussianprocess import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from Learner.GPTS_Learner import GPTS_Learner


class Experiment_2:
    def __init__(self, n_arms=10, sigma=10 ,price_env_id=0, adv_env_id=0):
        self.n_arms = n_arms
        self.n_rounds = 365
        min_bid = 0.0
        max_bid = 1.0

        self.bids = np.linspace(min_bid, max_bid, n_arms)
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

        # Experiment settings
        self.sigma = sigma

        self.opt_super_arm_reward = None  # self.run_clairvoyant()

        ## Rewards for each experiment (each element is a list of T rewards)
        self.opt_rewards_per_experiment = []
        self.gpts_rewards_per_experiment = []

    def run(self):
        cost_x_obs = np.array([])
        cost_y_obs = np.array([])

        nclick_x_obs = np.array([])
        nclick_y_obs = np.array([])

        convrate_x_obs = np.array([])
        convrate_y_obs = np.array([])

        dproba_x_obs = np.array([])
        dproba_y_obs = np.array([])

        cost_noise_std = 0.2
        nclick_noise_std = 10.0
        convrate_noise_std = 1.0
        dproba_noise_std = 1.0

        theta = 1.0
        l = 1.0
        kernel = C(theta, (1e-3, 1e3)) * RBF(l, (1e-3, 1e3))
        gps =[]
        for j in range(3):
            gps.append(GaussianProcessRegressor(kernel=kernel, alpha=cost_noise_std**2, normalize_y=False, n_restarts_optimizer=10))
            gps.append(GaussianProcessRegressor(kernel=kernel, alpha=nclick_noise_std**2, normalize_y=False, n_restarts_optimizer=10))
            gps.append(GaussianProcessRegressor(kernel=kernel, alpha=convrate_noise_std**2, normalize_y=False, n_restarts_optimizer=10))
            gps.append(GaussianProcessRegressor(kernel=kernel, alpha=dproba_noise_std**2, normalize_y=False, n_restarts_optimizer=10))
        for j, label in enumerate(self.feature_labels):
            for i in range(0, self.n_rounds-330):
                new_bid = np.random.choice(self.bids, 1)
                new_price = np.random.choice(self.prices, 1)

                new_cost_x_obs = new_bid
                new_cost_y_obs = generate_observation(self.cost_functions[label] ,new_cost_x_obs, cost_noise_std)

                cost_x_obs = np.append(cost_x_obs, new_cost_x_obs)
                cost_y_obs = np.append(cost_y_obs, new_cost_y_obs)

                X_cost = np.atleast_2d(cost_x_obs).T
                Y_cost = cost_y_obs.ravel()

                gps[j*4].fit(X_cost, Y_cost)


                new_nclick_x_obs = new_bid
                new_nclick_y_obs = generate_observation(self.click_functions[label] ,new_nclick_x_obs, nclick_noise_std)

                nclick_x_obs = np.append(nclick_x_obs, new_nclick_x_obs)
                nclick_y_obs = np.append(nclick_y_obs, new_nclick_y_obs)

                X_nclick = np.atleast_2d(nclick_x_obs).T
                Y_nclick = nclick_y_obs.ravel()

                gps[j*4+1].fit(X_nclick, Y_nclick)

                # new_convrate_x_obs = new_price
                # new_convrate_y_obs = generate_observation(self.demand_functions[j] ,new_convrate_x_obs, convrate_noise_std)

                # convrate_x_obs = np.append(convrate_x_obs, new_convrate_x_obs)
                # convrate_y_obs = np.append(convrate_y_obs, new_convrate_y_obs)

                # X_convrate = np.atleast_2d(convrate_x_obs).T
                # Y_convrate = convrate_y_obs.ravel()

                # gps[j*4+2].fit(X_convrate, Y_convrate)

                # new_dproba_x_obs = new_price
                # new_dproba_y_obs = generate_observation(self.future_visits[label] ,new_dproba_x_obs, dproba_noise_std)

                # dproba_x_obs = np.append(dproba_x_obs, new_dproba_x_obs)
                # dproba_y_obs = np.append(dproba_y_obs, new_dproba_y_obs)

                # X_dproba = np.atleast_2d(dproba_x_obs).T
                # Y_dproba = dproba_y_obs.ravel()

                # gps[j*4+3].fit(X_dproba, Y_dproba)



            x_pred_cost = np.atleast_2d(self.bids).T
            y_pred_cost, sigma_cost = gps[j*4].predict(x_pred_cost, return_std=True)    

            plt.figure(j*4)
            plt.plot(x_pred_cost, self.cost_functions[label](x_pred_cost), 'r:', label=r'$n(x)$')
            plt.plot(X_cost.ravel(), Y_cost, 'ro', label=u'Observed Cost per click')
            plt.plot(x_pred_cost, y_pred_cost, 'b-', label=u'Predicted Cost per click')
            plt.fill(np.concatenate([x_pred_cost, x_pred_cost[::-1]]),
                    np.concatenate([y_pred_cost - 1.96*sigma_cost, (y_pred_cost + 1.96*sigma_cost)[::-1]]),
                    alpha = .5, fc='b', ec='None', label = '95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc='lower right')
            plt.show()

            x_pred_nclick = np.atleast_2d(self.bids).T
            y_pred_nclick, sigma_nclick = gps[j*4+1].predict(x_pred_nclick, return_std=True)

            plt.figure(j*4+1)
            plt.plot(x_pred_nclick, self.click_functions[label](x_pred_nclick), 'r:', label=r'$n(x)$')
            plt.plot(X_nclick.ravel(), Y_nclick, 'ro', label=u'Observed Clicks')
            plt.plot(x_pred_nclick, y_pred_nclick, 'b-', label=u'Predicted Clicks')
            plt.fill(np.concatenate([x_pred_nclick, x_pred_nclick[::-1]]),
                    np.concatenate([y_pred_nclick - 1.96*sigma_nclick, (y_pred_nclick + 1.96*sigma_nclick)[::-1]]),
                    alpha = .5, fc='b', ec='None', label = '95% conf interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.legend(loc='lower right')
            plt.show()

            # x_pred_convrate = np.atleast_2d(self.prices).T
            # y_pred_convrate, sigma_convrate = gps[j*4+2].predict(x_pred_convrate, return_std=True)
            
            # plt.figure(j*4+2)
            # plt.plot(x_pred_convrate, self.cost_functions[label](x_pred_convrate), 'r:', label=r'$n(x)$')
            # plt.plot(X_convrate.ravel(), Y_convrate, 'ro', label=u'Observed Conversion rate')
            # plt.plot(x_pred_convrate, y_pred_convrate, 'b-', label=u'Predicted Conversion rate')
            # plt.fill(np.concatenate([x_pred_convrate, x_pred_convrate[::-1]]),
            #         np.concatenate([y_pred_convrate - 1.96*sigma_convrate, (y_pred_convrate + 1.96*sigma_nclick)[::-1]]),
            #         alpha = .5, fc='b', ec='None', label = '95% conf interval')
            # plt.xlabel('$x$')
            # plt.ylabel('$n(x)$')
            # plt.legend(loc='lower right')
            # plt.show()
            
            # x_pred_dproba = np.atleast_2d(self.prices).T
            # y_pred_dproba, sigma_dproba = gps[j*4+3].predict(x_pred_dproba, return_std=True)
            
            # plt.figure(j*4+3)
            # plt.plot(x_pred_dproba, self.cost_functions[label](x_pred_dproba), 'r:', label=r'$n(x)$')
            # plt.plot(X_dproba.ravel(), Y_dproba, 'ro', label=u'Observed Density probability')
            # plt.plot(x_pred_dproba, y_pred_dproba, 'b-', label=u'Predicted Density probability')
            # plt.fill(np.concatenate([x_pred_dproba, x_pred_dproba[::-1]]),
            #         np.concatenate([y_pred_dproba - 1.96*sigma_dproba, (y_pred_dproba + 1.96*sigma_nclick)[::-1]]),
            #         alpha = .5, fc='b', ec='None', label = '95% conf interval')
            # plt.xlabel('$x$')
            # plt.ylabel('$n(x)$')
            # plt.legend(loc='lower right')
            # plt.show()

            #Reinitialise for other categories of users
            cost_x_obs = np.array([])
            cost_y_obs = np.array([])

            nclick_x_obs = np.array([])
            nclick_y_obs = np.array([])

            convrate_x_obs = np.array([])
            convrate_y_obs = np.array([])

            dproba_x_obs = np.array([])
            dproba_y_obs = np.array([])
            
        
        return 


    # def run(self):


        
    #     """
    #     Optimization Problem Solution
    #     :return: list of optimal bid and price for each sub-campaign
    #     """
    #     N = len(self.feature_labels)

    #     table = [[] for row in range(N)]
    #     for j,label in enumerate(self.feature_labels):
    #         temp = [[0 for x in range(len(self.bids))] for y in range(len(self.prices))] 
    #         for k in range(len(self.prices)):
    #             for i in range(len(self.bids)):
    #                 ѵ = np.sum(self.n_visits * self.future_visits[label](self.n_visits))
    #                 v = (self.prices[k] * self.demand_functions[j][k]) * (1 + ѵ) - self.cost_functions[label](self.bids[i])
    #                 n = self.click_functions[label](self.bids[i])
    #                 temp[k][i] = v * n
    #         table[j] = temp

    #     opt_indexes = optimizer(table)

    #     return opt_indexes

def test():
    print("OK")
    return 0

test()