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
from scripts.test_env import StockEnvTest

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
TAUVALUE = 1
TIMEVALUE = 192021
TAULAYER = 256


TESTING_DATA_FILE = "test.csv"


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



def evaluate(frame, eval_runs=5, capture=True, render=False):
    """
    Makes an evaluation run
    """
    print("------------------------------------------EVALUATING---------------------------------------------------")
    print(frame)
    reward_batch = []
    frame_list = []
    all_scores = []
    run_number = []
    for i in range(eval_runs):
        state = eval_env.reset()
        #print(len(state))
        if render: eval_env.render()
        rewards = 0
        score = 0
        scores = []
        states6 = []
        states7 = []
        states8 = []
        states9 = []
        states10 = []
        actions = []
        rewards_list = []
        while True:
            action = agent.act(np.expand_dims(state, axis=0),add_noise=False)
            print(action)
            action_v = np.clip(action, action_low, action_high)
            #print(eval_env.day)
            state, reward, done, amnt_penalty = eval_env.step(action_v[0])
            rewards += reward
            score += reward
            scores.append(np.mean(score))
            reward_batch.append(rewards)

            #states6.append(state[6])
            #states7.append(state[7])
            #states8.append(state[8])
            #states9.append(state[9])
            #states10.append(state[10])
            #actions.append(action_v)
            #rewards_list.append(reward)
            if done:
                all_scores.append(np.mean(scores))
                frame_list.append(frame)
                run_number.append(i)
                #print(eval_env.day)
                #df = pd.DataFrame(list(zip(scores, actions, states6, states7, states8, states9, states10, rewards_list)))
                # print('mean of scores:{}'.format(np.mean(scores)))

                #df.to_csv('CSVs/score_state_actions_' + str(TAULAYER) + '_' + str(TIMEVALUE) + '_tau' + str(TAUVALUE) + '_eval.csv', mode='a', encoding='utf-8', index=False)
                break
        writer.add_scalar("Reward", np.mean(all_scores), frame)
            #print(reward_batch)

    df_scores = pd.DataFrame(list(zip(all_scores, frame_list, run_number)))
    df_scores.to_csv('CSVs/results_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_tau'+str(TAUVALUE)+'_eval_mean.csv', mode='a', encoding='utf-8', index=True)
    print('mean of scores:{}'.format(np.mean(df_scores)))


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
    action_list = []
    scores_list = []
    states_list = []
    episode_list = []
    time_stamp = 0
    for frame in range(1, frames + 1):
        # evaluation runs

        if frame % 100 == 0 :
            evaluate(frame*worker, eval_runs)

        #print(frame)

        action = agent.act(state)
        action_v = np.clip(action, action_low, action_high)
        next_state, reward, done, amnt_penalty = envs.step(action_v)

        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, frame, writer)

        if args.icm:
            reward_i = agent.icm.get_intrinsic_reward(state[0], next_state[0], action[0])
            curiosity_logs.append((frame, reward_i))

        score += reward
        scores_deque.append(score)
        scores.append(np.mean(score))

        average_100_scores.append(np.mean(scores_deque))

        if i_episode % 5 == 0:
            minmax_scores.append((np.min(scores_deque), np.max(scores_deque)))
            episode_list.append(i_episode)
            action_list.append(action)
            scores_list.append(np.mean(scores_deque))
            states_list.append(score)

            df = pd.DataFrame(list(zip(action_list,scores_list,states_list, episode_list,minmax_scores)))
            df.to_csv('results_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_tau_'+str(TAUVALUE)+'_.csv', mode='a', encoding='utf-8', index=False)
            torch.save(agent.actor_local.state_dict(), 'runs/checkpoint_actor_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_tau'+str(TAUVALUE)+'_' + str(i_episode) + ".pth")
            torch.save(agent.critic_local.state_dict(), 'runs/checkpoint_critic_'+str(TAULAYER)+'6_'+str(TIMEVALUE)+'_tau'+str(TAUVALUE)+'_' + str(i_episode) + ".pth")
            action_list = []
            scores_list = []
            states_list = []
            episode_list = []
        if i_episode % 2 == 0:
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                             np.mean(scores_window)), end="")
        state = next_state

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
parser.add_argument("-per", type=int, default=0, choices=[0, 1],
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
parser.add_argument("-frames", type=int, default=200000,
                    help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("-eval_every", type=int, default=500,
                    help="Number of interactions after which the evaluation runs are performed, default = 10000")
parser.add_argument("-eval_runs", type=int, default=5, help="Number of evaluation runs performed, default = 1")
parser.add_argument("-seed", type=int, default=3, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=3e-4,
                    help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-lr_c", type=float, default=3e-4,
                    help="Critic learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
parser.add_argument("-learn_number", type=int, default=1, help="Learn x times per interaction, default = 1")
parser.add_argument("-layer_size", type=int, default=TAULAYER,
                    help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6),
                    help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
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
from scripts.BasicEnvironment import BasicEnv
if __name__ == "__main__":

    preprocessed_path = "0001_test.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)

    data = data.drop(columns=["datadate_full"])

    data = data[["datadate","tic","Close","open","high","low","volume","macd","rsi","cci","adx"]]

    data.Close = data.Close.apply(np.int64)
    data.macd = data.macd.apply(np.int64)
    data.rsi = data.rsi.apply(np.int64)
    data.cci = data.cci.apply(np.int64)
    data.adx = data.adx.apply(np.int64)

    #print(data.to_string())
    train = data_split(data, start=20210101, end=20220701)
    test_d = data_split(data, start=20220701, end=20221001)

    unique_trade_date = data[(data.datadate > 20211010)].datadate.unique()
    print(len(unique_trade_date))

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
    eval_env = StockEnvTest(test_d)

    envs = MultiPro.SubprocVecEnv([lambda: BasicEnv() for i in range(args.worker)])
    eval_env = BasicEnv()
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
        for i in range(120,131,10):
            print("-------------------------------" + str(i) + "----------------------------")
            agent.actor_local.load_state_dict(torch.load('runs/checkpoint_actor_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_tau'+str(TAUVALUE)+'_'+str(i)+'.pth'))
            agent.critic_local.load_state_dict(torch.load('runs/checkpoint_critic_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_tau'+str(TAUVALUE)+'_'+str(i)+'.pth'))

            evaluate(frame=int(len(unique_trade_date))*worker, capture=True)
    else:
        run(frames=args.frames // args.worker,
            eval_every=args.eval_every // args.worker,
            eval_runs=args.eval_runs,
            worker=args.worker)

    t1 = time.time()
   # envs.close()
    timer(t0, t1)
    # save trained model
    torch.save(agent.actor_local.state_dict(), 'runs/evaluating_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_actor_tau?'+str(TAUVALUE)+'_' + args.info + ".pth")
    torch.save(agent.critic_local.state_dict(), 'runs/evaluating_'+str(TAULAYER)+'_'+str(TIMEVALUE)+'_critic_tau?'+str(TAUVALUE)+'_' + args.info + ".pth")
    # save parameter
    with open('runs/' + args.info + ".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

