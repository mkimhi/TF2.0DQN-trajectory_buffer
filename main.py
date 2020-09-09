import os
from time import time
import numpy as np
from config_utils import read_main_config
from q_learner import QLearning
from deep_q_network import DeepQNetwork
from gym_wrapper import GymWrapper
from summaries_collector import SummariesCollector
import pandas as pd
from matplotlib import pyplot as plt

#import logging

def run_q_learning(config,gym_wrapper,summaries_collector_traj,summaries_collector):
    trajectory_ratio = config['model']['trajectory_ratio']
    trajectory_ratio_2 = config['model']['trajectory_ratio_2']
    episodes_per_test = config['general']['episodes_per_test']
    map_size=config['general']['map_name']
    rewards = [[], []]
    losses = [np.zeros(5),np.zeros(5)] if (map_size=='4x4') else [np.zeros(14),np.zeros(14)]
    bum_to_avg=3
    for _ in range(bum_to_avg):
        q_learner = QLearning(config, gym_wrapper, trajectory_ratio=trajectory_ratio)
        initial_time = round(time(), 3)
        q_learner.train(summaries_collector_traj)
        trajectory_reward = q_learner.test(summaries_collector_traj, episodes=episodes_per_test*10, render=False)
        total_time_traj = round(time(), 3) - initial_time
        summaries_collector_traj.read_summaries('test')
        losses[0]=np.add(losses[0], q_learner.loss_evaluation())

        q_learner = QLearning(config, gym_wrapper, trajectory_ratio=trajectory_ratio_2)
        initial_time = round(time(), 3)
        q_learner.train(summaries_collector)
        no_trajectory_reward = q_learner.test(summaries_collector, episodes=episodes_per_test*10, render=False)
        total_time = round(time(), 3) - initial_time
        summaries_collector.read_summaries('test')
        losses[1] =np.add(losses[1], q_learner.loss_evaluation())

        rewards[0].append(trajectory_reward)
        rewards[1].append(no_trajectory_reward)
        print(
            "tested avg reward with regular buffer: {0}, with trajectory buffer: {1}".format(no_trajectory_reward,
                                                                                             trajectory_reward))
        print("total train and test time for no trajectory replay buffer: {0} seconds".format(total_time))
        print("total train and test time for trajectory replay buffer: {0} seconds".format(total_time_traj))
    print("-----summaries-----")
    print("avg reward for trajectory ratio {2} is: {0}, avg reward with trajectory ratio {3} is: {1}"
          .format(np.mean(rewards[0]),np.mean(rewards[1]),trajectory_ratio,trajectory_ratio_2))
    print("-------------------")

    losses = np.dot(losses,(1/bum_to_avg) ) #avg over the for loop
    plot_losses(losses)




def run_dqn(config,gym_wrapper,summaries_collector_traj,summaries_collector):
    q_network = DeepQNetwork(config, gym_wrapper,trajectory=1)
    initial_time = round(time(), 3)
    q_network.train(summaries_collector_traj)
    reward = q_network.test(summaries_collector_traj,episodes=100, render=True)
    summaries_collector_traj.read_summaries('test')
    total_time_traj = round(time(), 3) - initial_time
    print("tested avg reward: {0} ".format(reward))


def plot_losses(losses):
    df = pd.DataFrame({'Trajectory': losses[0], 'Regular': losses[1]}, columns=['Trajectory', 'Regular'])
    plt.figure()
    df.plot(kind='bar')
    plt.savefig('Loss by distance.png')
    df.plot(kind='pie', subplots=True)
    plt.savefig('pie losses by distance.png')
    df.plot(kind='box')
    plt.savefig('box losses by distance.png')


if __name__ == '__main__':
    config = read_main_config()
    gym_wrapper = GymWrapper(config)
    algorithm = config['general']['algorithm']
    summaries_dir = os.path.join(os.getcwd(), 'data', 'summaries')
    summaries_collector_traj = SummariesCollector(summaries_dir, 'Frozen_lake_Trajectory_buffer', config)
    summaries_collector_reg = SummariesCollector(summaries_dir, 'Frozen_lake_Regular_buffer', config)
    if (algorithm == 'DQN'):
        run_dqn(config,gym_wrapper,summaries_collector_traj,summaries_collector_reg)
    else:
        run_q_learning(config,gym_wrapper,summaries_collector_traj,summaries_collector_reg)
    
    """
    #dump log into a file
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('logger.log', 'a'))
    print = logger.info

    logging.shutdown()
    """