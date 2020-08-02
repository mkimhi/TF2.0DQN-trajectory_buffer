from config_utils import read_main_config
from deep_q_network import DeepQNetwork
from gym_wrapper import GymWrapper


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

config = read_main_config()
gym_wrapper = GymWrapper(config['general']['scenario'])
deep_q_network = DeepQNetwork(config, gym_wrapper)
deep_q_network.train()
print("==================== TEST TIME: ====================")
deep_q_network.test(episodes=3,render=True)