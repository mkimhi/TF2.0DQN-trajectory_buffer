import gym

class GymWrapper:
    def __init__(self, env_string):
        self.env_string = env_string
        self._num_actions = None
        self._state_size = None

    def get_env(self):
        env = gym.make(self.env_string,is_slippery=True)
        return env

    def _set_data(self):
        env = self.get_env()
        self._num_actions = env.action_space.n
        if(self.env_string == 'FrozenLake-v0'):
            self._state_size = (1,)
        else:
            self._state_size = env.observation_space.shape
        env.close()

    def get_num_actions(self):
        if self._num_actions is None:
            self._set_data()
        return self._num_actions

    def get_state_size(self):
        if self._state_size is None:
            self._set_data()
        return self._state_size
