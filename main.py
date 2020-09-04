import os
from time import time
import numpy as np
from config_utils import read_main_config
from q_learner import QLearning
from deep_q_network import DeepQNetwork
from gym_wrapper import GymWrapper
from summaries_collector import SummariesCollector
import logging

def run_q_learning(config,gym_wrapper,summaries_collector_traj,summaries_collector):
    q_learner = QLearning(config, gym_wrapper, trajectory=True)
    initial_time = round(time(), 3)
    q_learner.train(summaries_collector_traj)
    trajectory_test_reward = q_learner.test(summaries_collector_traj, episodes=100, render=True)
    total_time_traj = round(time(), 3) - initial_time
    summaries_collector_traj.read_summaries('test')
    q_learner.loss_evaluation()


    q_learner = QLearning(config, gym_wrapper, trajectory=False)
    initial_time = round(time(), 3)
    q_learner.train(summaries_collector)
    no_trajectory_test_reward = q_learner.test(summaries_collector, episodes=100, render=True)
    total_time = round(time(), 3) - initial_time
    summaries_collector.read_summaries('test')
    q_learner.loss_evaluation()

    print("tested avg reward with regular buffer: {0}, with trajectory buffer: {1}".format(no_trajectory_test_reward,
                                                                                           trajectory_test_reward))
    print("total train and test time for no trajectory replay buffer: {0} seconds".format(total_time))
    print("total train and test time for trajectory replay buffer: {0} seconds".format(total_time_traj))
    return trajectory_test_reward,no_trajectory_test_reward

def run_dqn(config,gym_wrapper,summaries_collector_traj,summaries_collector):
    q_network = DeepQNetwork(config, gym_wrapper,trajectory=True)
    initial_time = round(time(), 3)
    q_network.train(summaries_collector_traj)
    reward = q_network.test(summaries_collector_traj,episodes=100, render=True)
    summaries_collector_traj.read_summaries('test')
    total_time_traj = round(time(), 3) - initial_time
    print("tested avg reward: {0} ".format(reward))

if __name__ == '__main__':
    config = read_main_config()
    gym_wrapper = GymWrapper(config['general']['scenario'])
    algorithm = config['general']['algorithm']
    summaries_dir = os.path.join(os.getcwd(), 'data', 'summaries')
    summaries_collector_traj = SummariesCollector(summaries_dir, 'Frozen_lake_Trajectory_buffer', config)
    summaries_collector_reg = SummariesCollector(summaries_dir, 'Frozen_lake_Regular_buffer', config)
    if (algorithm == 'DQN'):
        run_dqn(config,gym_wrapper,summaries_collector_traj,summaries_collector_reg)
    else:
        rewards = [[],[]]
        for i in range(3):
            trajectory_reward, reg_reward= run_q_learning(config,gym_wrapper,summaries_collector_traj,summaries_collector_reg)
            rewards[0].append(trajectory_reward)
            rewards[1].append(reg_reward)
        print("avg reward for trajectory is: {0}, avg reward without trajectory is: {1}".format(np.mean(rewards[0]),np.mean(rewards[1])))
    """
    #dump log into a file
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('logger.log', 'a'))
    print = logger.info

    logging.shutdown()
    """