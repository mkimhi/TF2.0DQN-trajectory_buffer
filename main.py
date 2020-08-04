from config_utils import read_main_config
from q_learner import QLearning
from gym_wrapper import GymWrapper
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

config = read_main_config()
gym_wrapper = GymWrapper(config['general']['scenario'])
q_learner = QLearning(config, gym_wrapper)
q_learner.train()
print("==================== TEST TIME: ====================")
q_learner.test(episodes=3,render=True)