import numpy as np
import random
import gym
from collections import deque
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
# from  files import MultiPro
from scripts.agent import Agent
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts import MultiPro
import json
from scripts.environment import StockEnvTrain

import pandas as pd

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

import datetime

import datetime
import os

TRAINING_DATA_FILE = "dataprocessing/Yfinance_Data.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)

TESTING_DATA_FILE = "test.csv"


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)

    return _data


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)

    # data  = data[final_columns]
    data.index = data.datadate.factorize()[0]

    return data


def calculate_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()

    data = data[['Date', 'tic', 'Close', 'Open', 'High', 'Low', 'Volume', 'datadate']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data


def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = Sdf.retype(df.copy())

    # print(stock)

    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    # temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
        ## macd
        temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
        temp_macd = pd.DataFrame(temp_macd)
        macd = macd.append(temp_macd, ignore_index=True)
        ## rsi
        temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = rsi.append(temp_rsi, ignore_index=True)
        ## cci
        temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
        temp_cci = pd.DataFrame(temp_cci)
        cci = cci.append(temp_cci, ignore_index=True)
        ## adx
        temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
        temp_dx = pd.DataFrame(temp_dx)
        dx = dx.append(temp_dx, ignore_index=True)

    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df


def preprocess_data():
    """data preprocessing pipeline"""
    start = datetime.datetime(2010, 12, 1)
    df = load_dataset(file_name=TRAINING_DATA_FILE)
    # get data after 2010
    # df = df[df.Date >= start]
    # calcualte adjusted price
    df_preprocess = calculate_price(df)
    # add technical indicators using stockstats
    df_final = add_technical_indicator(df_preprocess)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill', inplace=True)
    return df_final

def evaluate(frame, eval_runs=5, capture=True, render=False):
    """
    Makes an evaluation run
    """

    reward_batch = []

    all_scores = []
    for i in range(eval_runs):
        state = eval_env.reset()
        if render: eval_env.render()
        rewards = 0
        score = 0
        scores = []
        states = []
        actions = []
        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)

            state, reward, done, amnt_penalty = eval_env.step(action_v[0])
            rewards += reward
            score += reward
            scores.append(np.mean(score))
            states.append(states)
            actions.append(action_v)
            if done:
                break
        reward_batch.append(rewards)
        if capture == True:
            all_scores.append(np.mean(scores))
            df = pd.DataFrame(list(zip(scores, state, actions)))
            #print('mean of scores:{}'.format(np.mean(scores)))
            df.to_csv('results_128_eval.csv', mode='a', encoding='utf-8', index=False)
            writer.add_scalar("Reward", np.mean(reward_batch), frame)
            #print(reward_batch)
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv('results_128_eval_mean.csv', mode='a', encoding='utf-8', index=False)
    print('mean of scores:{}'.format(np.mean(df_scores)))


# The algorithms require a vectorized environment to run
def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def run(frames=1000, eval_every=10000, eval_runs=5, worker=1):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []
    amount_penalty = []
    # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    i_episode = 1
    state = envs.reset()
    score = 0
    curiosity_logs = []
    scores_deque = deque(maxlen=100)
    scores = []
    minmax_scores = []
    average_100_scores = []
    episodes = []
    time_stamp = 0
    for frame in range(1, frames + 1):
        # evaluation runs

        if frame % eval_every == 0 or frame == 550:
            evaluate(frame*worker, eval_runs)


        action = agent.act(state)
        action_v = np.clip(action, action_low, action_high)
        next_state, reward, done, amnt_penalty = envs.step(action_v)

        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, frame, writer)

        if args.icm:
            reward_i = agent.icm.get_intrinsic_reward(state[0], next_state[0], action[0])
            curiosity_logs.append((frame, reward_i))
        state = next_state
        score += reward

        amount_penalty.append(amnt_penalty)
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        minmax_scores.append((np.min(score), np.max(score)))
        average_100_scores.append(np.mean(scores_deque))
        episodes.append(i_episode)

        if i_episode % 10 == 0:
            df = pd.DataFrame(list(zip(episodes,average_100_scores, scores_deque, amount_penalty, state, action_v)))
            df.to_csv('results_128_2016_2019.csv', mode='a', encoding='utf-8', index=False)
            torch.save(agent.actor_local.state_dict(), "runs/checkpoint_actor_128_2016_2019_" + str(i_episode) + ".pth")
            torch.save(agent.critic_local.state_dict(), "runs/checkpoint_critic_128_2016_2019_" + str(i_episode) + ".pth")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                             np.mean(scores_window)), end="")

        if done.any():
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame * worker)
            for v in curiosity_logs:
                i, r = v[0], v[1]
                writer.add_scalar("Intrinsic Reward", r, i)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                             np.mean(scores_window)), end="")
            # if i_episode % 100 == 0:
            #    print('\rEpisode {}\tFrame \tReward: {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, round(eval_reward,2), np.mean(scores_window)), end="", flush=True)
            i_episode += 1
            state = envs.reset()
            score = 0
            curiosity_logs = []


parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str, default="Pendulum-v0", help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("--device", type=str, default="gpu", help="Select trainig device [gpu/cpu], default = gpu")
parser.add_argument("-nstep", type=int, default=1, help="Nstep bootstrapping, default 1")
parser.add_argument("-per", type=int, default=1, choices=[0, 1],
                    help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("-munchausen", type=int, default=0, choices=[0, 1],
                    help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("-iqn", type=int, choices=[0, 1], default=1,
                    help="Use distributional IQN Critic if set to 1, default = 1")
parser.add_argument("-noise", type=str, choices=["ou", "gauss"], default="OU",
                    help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")
parser.add_argument("-info", type=str, default="runsfirst", help="Information or name of the run")
parser.add_argument("-d2rl", type=int, choices=[0, 1], default=0,
                    help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("-frames", type=int, default=100000,
                    help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("-eval_every", type=int, default=10000,
                    help="Number of interactions after which the evaluation runs are performed, default = 10000")
parser.add_argument("-eval_runs", type=int, default=1, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=3e-4,
                    help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-lr_c", type=float, default=3e-4,
                    help="Critic learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
parser.add_argument("-learn_number", type=int, default=1, help="Learn x times per interaction, default = 1")
parser.add_argument("-layer_size", type=int, default=128,
                    help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6),
                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-3,
                    help="Softupdate factor tau, default is 1e-3")  # for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("-w", "--worker", type=int, default=2, help="Number of parallel environments, default = 1")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--icm", type=int, default=0, choices=[0, 1],
                    help="Using Intrinsic Curiosity Module, default=0 (NO!)")
parser.add_argument("--add_ir", type=int, default=0, choices=[0, 1],
                    help="Add intrisic reward to the extrinsic reward, default = 0 (NO!) ")

args = parser.parse_args()

if __name__ == "__main__":

    preprocessed_path = "done_3stocks.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)

    unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200808)].datadate.unique()
    # print(unique_trade_date)

    train = data_split(data, start=20160101, end=20190101)
    test_d = data_split(data, start=20190101, end=20200101)

    env_name = args.env
    seed = args.seed
    frames = args.frames
    worker = args.worker
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size * args.worker
    LR_ACTOR = args.lr_a  # learning rate of the actor
    LR_CRITIC = args.lr_c  # learning rate of the critic
    saved_model = args.saved_model
    D2RL = args.d2rl

    writer = SummaryWriter("runs/" + args.info)

    #envs = DummyVecEnv([lambda: StockEnvTrain(train)])
    #envs = MultiPro.SubprocVecEnv([lambda: gym.make(args.env) for i in range(args.worker)])
    envs = MultiPro.SubprocVecEnv([lambda: StockEnvTrain(train) for i in range(args.worker)])
    eval_env = StockEnvTrain(test_d)
    envs.seed(seed)
    eval_env.seed(seed+1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        "CUDA is not available"
        device = torch.device("cpu")
    print("Using device: {}".format(device))

    action_high = envs.action_space.high[0]
    action_low = envs.action_space.low[0]
    state_size = envs.observation_space.shape[0]
    action_size = envs.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, n_step=args.nstep, per=args.per,
                  munchausen=args.munchausen, distributional=args.iqn,
                  D2RL=D2RL, curiosity=(args.icm, args.add_ir), noise_type=args.noise, random_seed=seed,
                  hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                  LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, TAU=TAU, LEARN_EVERY=args.learn_every,
                  LEARN_NUMBER=args.learn_number, device=device, frames=args.frames, worker=args.worker)

    t0 = time.time()
    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load('runs/checkpoint_actor_128100.pth'))
        agent.critic_local.load_state_dict(torch.load('runs/checkpoint_critic_128100.pth'))
        evaluate(frame=500, capture=True)
    else:
        run(frames=args.frames // args.worker,
            eval_every=args.eval_every // args.worker,
            eval_runs=args.eval_runs,
            worker=args.worker)

    t1 = time.time()
   # envs.close()
    timer(t0, t1)
    # save trained model 
    torch.save(agent.actor_local.state_dict(), 'runs/evaluating_2016_2019_actor' + args.info + ".pth")
    torch.save(agent.critic_local.state_dict(), 'runs/evaluating_2016_2019_critic' + args.info + ".pth")
    # save parameter
    with open('runs/' + args.info + ".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)