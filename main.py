import os
from config_utils import read_main_config
from q_learner import QLearning
from gym_wrapper import GymWrapper
from summaries_collector import SummariesCollector
from tensorflow.python.framework.ops import disable_eager_execution


def create_summeries_dir(name):
    working_dir = os.getcwd()
    return os.path.join(working_dir, 'tensorboard', name)


if __name__ == '__main__':
    disable_eager_execution()
    config = read_main_config()
    gym_wrapper = GymWrapper(config['general']['scenario'])
    summaries_dir = create_summeries_dir('Frozen_lake_Trajectory_buffer')
    summaries_collector = SummariesCollector(summaries_dir, 'Frozen_lake_Trajectory_buffer', config)
    q_learner = QLearning(config, gym_wrapper)
    q_learner.train(summaries_collector)
    q_learner.test(summaries_collector,episodes=50,render=True)






    #summaries_dir = create_summeries_dir('Frozen_lake_Regular_buffer')




