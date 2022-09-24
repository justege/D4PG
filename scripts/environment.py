import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
import os
from gym import spaces
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 10
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 1000
# total number of stocks in our portfolio
STOCK_DIM = 3
# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4


class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.day = day
        self.df = df
        self.agent_stock_iteration_index = 0
        self.penalty = 0

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
        # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(19,))
        # load data from a pandas dataframe
        # print('df: {}'.format(self.df))
        # print('day: {}'.format(self.day))
        self.data = self.df.loc[self.day, :]
        self.amount_penalty = 0
        self.terminal = False

        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.Close.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.final_asset_value = 0
        self.trades = 0
        self.previous_trades = 0
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        action = action
        # print('index:{}'.format(index))
        # print('selling:{}'.format(action))
        # perform sell action based on the sign of the action
        if self.state[index+STOCK_DIM+1] > 0:
            #update balance
            self.state[0] += \
            self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             (1- TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
             TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    def _buy_stock(self, index, action):
        # print('index:{}'.format(index))
        # print('buying:{}'.format(action))
        action = action
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))

        # update balance
        self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                         (1 + TRANSACTION_FEE_PERCENT)

        self.state[index + STOCK_DIM + 1] += min(available_amount, action)

        self.cost += self.state[index + 1] * min(available_amount, action) * \
                     TRANSACTION_FEE_PERCENT

        if available_amount > 0:
            self.trades += 1

    def step(self, actions):
        # print(self.day)
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)
        self.actions = actions
        if self.terminal:
            print("Finished")
            print(self.state)
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

            print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            # df_total_value.to_csv('results/account_value_train.csv')
            # print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            # print("total_cost: ", self.cost)
            # print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                     df_total_value['daily_return'].std()
            print("Sharpe: ", sharpe)
            # print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            # df_rewards.to_csv('results/account_rewards_train.csv')

            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            # with open('obs.pkl', 'wb') as f:
            #    pickle.dump(self.state, f)

            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))
            # print("The actions is: {}".format(self.actions))

            # action = np.array([4,4,5])
            # actions = np.array([4,0,0,0,0,0,0,0,4,0,4,0,-3,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0])

            # actions = self.actions * HMAX_NORMALIZE #WHY??
            # print("actions-index------:{}".format(actions))
            # actions = (actions.astype(int))
            # print(actions)
            actions = actions * HMAX_NORMALIZE
            # print('all-actions:{}'.format(actions))

            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                    self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)  # TODO: this may not be touched.
            # print("The actions is: {}".format(actions))

            sell_index = argsort_actions[:np.where(actions < 0.5)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > -0.5)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            # print("self.day:{}".format(self.day))
            # --print('trades:{}'.format(self.trades))

            if self.previous_trades == self.trades:
                self.amount_penalty = +1
                self.penalty = 10
                # self.reward = -1
            else:
                self.penalty = 0
                # self.reward = self.trades

            self.previous_trades = float(self.trades)

            # load next state
            # print("stock_shares:{}".format(self.state[29:]))
            self.state =  [self.state[0]] + \
                self.data.Close.values.tolist() + \
                list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist()


            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
                                  self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))
            self.asset_memory.append(end_total_asset)
            # print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset - self.penalty

            # print("trades:{}".format(self.trades))
            # print('previous:{}'.format(self.previous_trades))
            # print("penalty:{}".format(self.penalty))
            # print("step_reward:{}".format(self.reward))

            self.rewards_memory.append(self.reward)
            # self.reward = self.reward*REWARD_SCALING
            # print("step_reward:{}".format(self.reward))

            self.day += 1
            self.data = self.df.loc[self.day, :]

        return self.state, self.reward, self.terminal, self.amount_penalty

    def reset(self):
        self.amount_penalty = 0
        self.final_asset_value = 0
        self.trades = 0
        self.previous_trades = 0
        self.penalty = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.agent_stock_iteration_index = 0
        # initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.Close.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        # iteration += 1
        # print("[0]*STOCK_DIM:{}".format([0]*STOCK_DIM))
        # print("self.state:{}".format(len(self.state)))
        # print(np.array(self.state))
        return np.array(self.state)

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

