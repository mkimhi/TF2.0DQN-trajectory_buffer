import numpy as np

from episode_collector import EpisodeCollector
from replay_buffer import ReplayBuffer


class QLearning:
    def __init__(self, config, gym_wrapper):
        self.config = config
        self.gym_wrapper = gym_wrapper
        env = self.gym_wrapper.get_env()
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.replay_buffer = ReplayBuffer(self.config['model']['replay_buffer_size'])

    def train(self):
        env = self.gym_wrapper.get_env()
        completion_reward = self.config['general']['completion_reward']
        epsilon = self.config['model']['epsilon']
        episode_collector = EpisodeCollector(self.q_table, env, self.gym_wrapper.get_num_actions())
        for cycle in range(self.config['general']['cycles']):
            #print('cycle {} epsilon {}'.format(cycle, epsilon))
            epsilon, train_avg_reward = self._train_cycle(episode_collector, epsilon)
            if (cycle + 1) % self.config['general']['test_frequency'] == 0 or (completion_reward is not None and train_avg_reward > completion_reward):
                test_avg_reward = self.test(False)
                if completion_reward is not None and test_avg_reward > completion_reward:
                    print('TEST avg reward {} > required reward {}... stopping training'.format(test_avg_reward, completion_reward))
                    break
        env.close()

    def _train_cycle(self, episode_collector, epsilon):
        # collect data
        max_episode_steps = self.config['general']['max_train_episode_steps']
        rewards_per_episode = []
        for _ in range(self.config['general']['episodes_per_training_cycle']):
            states, actions, rewards, is_terminal_flags = episode_collector.collect_episode(
                max_episode_steps, epsilon=epsilon)
            self.replay_buffer.add_episode(states, actions, rewards, is_terminal_flags)
            rewards_per_episode.append(sum(rewards))
        avg_rewards = np.mean(rewards_per_episode)
        if (avg_rewards > 0.2 ):
            print('collected rewards: {}'.format(avg_rewards))
        epsilon *= self.config['model']['decrease_epsilon']
        epsilon = max(epsilon, self.config['model']['min_epsilon'])

        # train steps
        for _ in range(self.config['model']['train_steps_per_cycle']):
            self._train_step()
        return epsilon, avg_rewards

    def _train_step(self):
        batch_size = self.config['model']['batch_size']
        gamma = self.config['general']['gamma']
        lr = self.config['policy_network']['learn_rate']

        current_state, action, reward, next_state, is_terminal = zip(*self.replay_buffer.sample_batch(batch_size))
        for i in range(batch_size):
            next_q_values = self.q_table[next_state[i]]
            max_next_q_value = np.max(next_q_values, axis=-1)
            chosen_action=action[i]
            #chosen_action = np.argmax(self.q_table[current_state[i]])
            #bellman equation
            q_label = np.array(reward[i]) + gamma*(1. - is_terminal[i]) * max_next_q_value
            curr_q_label = self.q_table[current_state[i], chosen_action]
            self.q_table[np.array(current_state),chosen_action] =curr_q_label*(1-lr) + lr* q_label



    def test(self, render=True, episodes=None):
        env = self.gym_wrapper.get_env()
        episode_collector = EpisodeCollector(self.q_table, env, self.gym_wrapper.get_num_actions())
        max_episode_steps = self.config['general']['max_test_episode_steps']
        rewards_per_episode = []
        if episodes is None:
            episodes = self.config['general']['episodes_per_test']
        for _ in range(episodes):
            rewards = episode_collector.collect_episode(max_episode_steps, epsilon=0., render=render)[2]
            rewards_per_episode.append(sum(rewards))
        env.close()
        print(self.q_table)
        avg_reward = np.mean(rewards_per_episode)
        print('TEST collected rewards: {}'.format(avg_reward))
        return avg_reward