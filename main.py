import os
from time import time
from config_utils import read_main_config
from q_learner import QLearning
from gym_wrapper import GymWrapper
from summaries_collector import SummariesCollector

def create_summeries_dir(name):
    working_dir = os.getcwd()
    return os.path.join(working_dir, name)


if __name__ == '__main__':
    config = read_main_config()
    gym_wrapper = GymWrapper(config['general']['scenario'])
    summaries_dir = create_summeries_dir('Frozen_lake_Trajectory_buffer')
    summaries_collector = SummariesCollector(summaries_dir, 'Frozen_lake_Trajectory_buffer', config)
    q_learner = QLearning(config, gym_wrapper,trajectory = True)
    initial_time = round(time(), 3)
    q_learner.train(summaries_collector)
    q_learner.test(summaries_collector,episodes=50,render=True)
    summaries_collector.read_summaries('test')
    total_time = round(time(),3)-initial_time
    print("total train and test time for trajectory replay buffer: {0} seconds".format(total_time))

    summaries_dir = create_summeries_dir('Frozen_lake_Regular_buffer')
    summaries_collector = SummariesCollector(summaries_dir, 'Frozen_lake_Regular_buffer', config)
    q_learner = QLearning(config, gym_wrapper, trajectory=False)
    initial_time = round(time(), 3)
    q_learner.train(summaries_collector)
    q_learner.test(summaries_collector, episodes=50, render=True)
    summaries_collector.read_summaries('test')
    total_time = round(time(), 3) - initial_time
    print("total train and test time for no trajectory replay buffer: {0} seconds".format(total_time))


