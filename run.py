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
from scripts.environment import StockEnvTrainWithTA
from scripts.EnvironmentVersion3.environmentVersion3 import StockEnvTrainVersion3
from scripts.EnvironmentWithPV.ValidateEnvironmentWithPV import StockEnvValidateWithPV
from scripts.EnvironmentWithPV.environmentWithPV import StockEnvTrainWithPV
from scripts.EnvironmentWithoutPV.test_env import StockEnvTest
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


MAX = 500000

TRAINED = None

TAU = 1

COMMENT = 'DistributedAlgoVersion3_NewData_' + str(TAU) + '_Tau' + str(MAX)


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

def evaluate(env, eval_runs=1, render=False):
    """
    Makes an evaluation run
    """

    print("------------------------------------------EVALUATING---------------------------------------------------")
    eval_env = env
    for i in range(eval_runs):
        state = eval_env.reset()
        if render: eval_env.render()
        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)
            state, reward, done, info = eval_env.step(action_v[0])
            print(info)
            if done:
                break


# The algorithms require a vectorized environment to run
def timer(start, end):
    """ Helper to print training time """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def run(frames=1000, eval_every=1000, eval_runs=5, worker=1):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    i_episode = 1
    state = envs.reset()
    score = 0
    curiosity_logs = []
    scores = []
    for frame in range(1, frames + 1):
        # evaluation runs

        if frame % eval_every == 0 or frame == 2000:
            evaluate(train_env)

        action = agent.act(state)
        action_v = np.clip(action, action_low, action_high)
        next_state, reward, done, info = envs.step(action_v)

        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, frame, writer)

        if args.icm:
            reward_i = agent.icm.get_intrinsic_reward(state[0], next_state[0], action[0])
            curiosity_logs.append((frame, reward_i))
        state = next_state
        score += reward

        if i_episode % 30 == 0:
            if TRAINED != None:
                PATH = "runs/model" + COMMENT + str(i_episode + TRAINED) + ".pt"
            else:
                PATH = "runs/model" + COMMENT + str(i_episode) + ".pt"


            torch.save({
                'epoch': frame,
                'actor_model_state_dict': agent.actor_local.state_dict(),
                'critic_model_state_dict':agent.critic_local.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            }, PATH)

        if done.any():
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), frame * worker)

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
parser.add_argument("-noise", type=str, choices=["ou", "gauss"], default="gauss",
                    help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")
parser.add_argument("-info", type=str, default="runsfirst", help="Information or name of the run")
parser.add_argument("-d2rl", type=int, choices=[0, 1], default=0,
                    help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("-frames", type=int, default=MAX,
                    help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("-eval_every", type=int, default=2000,
                    help="Number of interactions after which the evaluation runs are performed, default = 10000")
parser.add_argument("-eval_runs", type=int, default=1, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=3, help="Seed for the env and torch network weights, default is 0")
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
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-3,
                    help="Softupdate factor tau, default is 1e-3")  # for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel environments, default = 1")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--icm", type=int, default=0, choices=[0, 1],
                    help="Using Intrinsic Curiosity Module, default=0 (NO!)")
parser.add_argument("--add_ir", type=int, default=0, choices=[0, 1],
                    help="Add intrisic reward to the extrinsic reward, default = 0 (NO!) ")


args = parser.parse_args()

if __name__ == "__main__":

    preprocessed_path = "0001_test.csv"


    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)

    unique_trade_date = data[(data.datadate > 20150101) & (data.datadate <= 20230101)].datadate.unique()
    # print(unique_trade_date)

    data = data[["datadate", "tic", "adjcp", "open", "high", "low", "volume", "macd", "rsi", "cci", "adx"]]

    data['adjcp'] = round(data['adjcp'], 1)
    data['macd'] = round(data['macd'], 1)
    data['rsi'] = round(data['rsi'], 1)
    data['cci'] = round(data['cci'], 1)
    data['adx'] = round(data['adx'], 1)

    train = data_split(data, start=20150101, end=20200101)
    test_d = data_split(data, start=20200101, end=20210101)

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
    envs = MultiPro.SubprocVecEnv([lambda: StockEnvTrainVersion3(train) for i in range(args.worker)])
    train_env = StockEnvTrainVersion3(train)
    train_env.seed(seed)
    envs.seed(seed)

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


    if TRAINED != None:
        print('OK, i will load the already trained models and opitimizers for you...')
        checkpoint = torch.load(
            "/Users/egemenokur/PycharmProjects/D4PG_New_season/runs/model" + COMMENT + str(TRAINED) + ".pt")
        agent.actor_local.load_state_dict(checkpoint['actor_model_state_dict'])
        agent.critic_local.load_state_dict(checkpoint['critic_model_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        print(checkpoint)

    run(frames=args.frames // args.worker,
        eval_every=args.eval_every // args.worker,
        eval_runs=args.eval_runs,
        worker=args.worker)

    t1 = time.time()
   # envs.close()
    timer(t0, t1)
    # save trained model
    # save parameter
    with open('runs/' + args.info + ".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)