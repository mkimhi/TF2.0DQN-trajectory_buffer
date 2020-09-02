import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl

from episode_collector import EpisodeCollector
from replay_buffer import ReplayBuffer
from trajectory_replay_buffer import TrajectoryReplayBuffer

class DeepQNetwork:
    def __init__(self, config, gym_wrapper,trajectory=True):
        self.config = config
        self.gym_wrapper = gym_wrapper
        self.q_model = self._create_net()
        #show arch
        #dot_img_file = './model_1.png'
        #tf.keras.utils.plot_model(self.q_model, to_file=dot_img_file, show_shapes=True)
        self.q_target_model = self._create_net()
        buf_size = self.config['model']['replay_buffer_size']
        self.replay_buffer = TrajectoryReplayBuffer(buf_size) if trajectory else ReplayBuffer(buf_size)

    def _create_net(self):
        activation = self.config['policy_network']['activation']
        model = keras.Sequential()
        for i, l in enumerate(self.config['policy_network']['layers']):
            if i == 0:
                state_size= self.gym_wrapper.get_state_size()
                model.add(kl.Dense(128, activation=activation, input_shape=state_size))
            else:
                model.add(kl.Dense(128, activation=activation))
        model.add(kl.Dense(self.gym_wrapper.get_num_actions()))

        model.compile(
            optimizer=ko.Adam(lr=self.config['policy_network']['learn_rate']),
            loss=[self._get_mse_for_action]
        )
        return model

    def _get_mse_for_action(self, target_and_action, current_prediction):
        targets, one_hot_action = tf.split(target_and_action, [1, self.gym_wrapper.get_num_actions()], axis=1)
        active_q_value = tf.expand_dims(tf.reduce_sum(current_prediction * one_hot_action, axis=1), axis=-1)
        return kls.mean_squared_error(targets, active_q_value)

    def _update_target(self):
        q_weights = self.q_model.get_weights()
        q_target_weights = self.q_target_model.get_weights()

        tau = self.config['policy_network']['tau']
        q_weights = [tau * w for w in q_weights]
        q_target_weights = [(1. - tau) * w for w in q_target_weights]
        new_weights = [
            q_weights[i] + q_target_weights[i]
            for i in range(len(q_weights))
        ]
        self.q_target_model.set_weights(new_weights)

    def _one_hot_action(self, actions):
        action_index = np.array(actions)
        batch_size = len(actions)
        result = np.zeros((batch_size, self.gym_wrapper.get_num_actions()))
        result[np.arange(batch_size), action_index] = 1.
        return result

    def train(self,summaries_collector):
        env = self.gym_wrapper.get_env()
        #q_table = numpy.zeros((env.observation_space.n, env.action_space.n))
        completion_reward = self.config['general']['completion_reward']
        episode_collector = EpisodeCollector(self.q_model, env, self.gym_wrapper.get_num_actions(),is_deep=True)
        epsilon = self.config['model']['epsilon']

        reward = 0
        episodes_len = 0
        loss = 0
        for cycle in range(self.config['general']['cycles']):
            print('cycle {} epsilon {}'.format(cycle, epsilon))
            epsilon, cycle_reward, cycle_episode_len, cycle_loss = self._train_cycle(episode_collector, epsilon)#####
            reward += cycle_reward
            episodes_len += cycle_episode_len
            loss += cycle_loss
            # save to csv
            if not cycle % 100:
                loss = loss / 100
                reward = reward / 100
                episodes_len = episodes_len / 100
                summaries_collector.write_summaries('train', cycle, reward, episodes_len, loss)
                reward = 0
                episodes_len = 0
                loss = 0

            # read once in 1000 episodes and plot into an image
            if (cycle % 1000 == 0):
                summaries_collector.read_summaries('train')
                summaries_collector.read_summaries('test')


            if (cycle + 1) % self.config['general']['test_frequency'] == 0 or (reward > completion_reward):
                test_avg_reward = self.test(False)
                if completion_reward is not None and test_avg_reward > completion_reward:
                    print('TEST avg reward {} > required reward {}... stopping training'.format(
                        test_avg_reward, completion_reward))
                    break
        env.close()

    def _train_cycle(self, episode_collector, epsilon):
        # collect data
        max_episode_steps = self.config['general']['max_train_episode_steps']
        rewards_per_episode = []
        cycle_len = 0
        for _ in range(self.config['general']['episodes_per_training_cycle']):
            states, actions, rewards, is_terminal_flags = episode_collector.collect_episode(
                max_episode_steps, epsilon=epsilon)
            self.replay_buffer.add_episode(states, actions, rewards, is_terminal_flags)
            rewards_per_episode.append(sum(rewards))
            cycle_len += len(actions)


        avg_rewards = np.mean(rewards_per_episode)
        print('collected rewards: {}'.format(avg_rewards))
        epsilon *= self.config['model']['decrease_epsilon']
        epsilon = max(epsilon, self.config['model']['min_epsilon'])
        steps = self.config['model']['train_steps_per_cycle']
        # train steps
        cycle_loss=0
        for _ in range(steps):
            cycle_loss +=self._train_step()
        # update target network
        self._update_target()
        episodes = self.config['general']['episodes_per_training_cycle']
        return epsilon, avg_rewards, cycle_len/episodes,cycle_loss/steps

    def _train_step(self):
        batch_size = self.config['model']['batch_size']
        current_state, action, reward, next_state, is_terminal = zip(*self.replay_buffer.sample_batch(batch_size))
        next_q_values = self.q_target_model.predict(np.array(next_state))
        max_next_q_value = np.max(next_q_values, axis=-1)
        target_labels = np.array(reward) + (1. - np.array(is_terminal)) * max_next_q_value
        one_hot_actions = self._one_hot_action(action)
        target_and_actions = np.concatenate((target_labels[:, None], one_hot_actions), axis=1)
        loss = self.q_model.train_on_batch(np.array(current_state), target_and_actions)
        return loss

    def test(self,summaries_collector, render=True, episodes=None):
        ##test_on_batch
        env = self.gym_wrapper.get_env()
        episode_collector = EpisodeCollector(self.q_model, env, self.gym_wrapper.get_num_actions(),is_deep=True)
        max_episode_steps = self.config['general']['max_test_episode_steps']
        if episodes is None:
            episodes = self.config['general']['episodes_per_test']
        episode_len = 0
        reward = 0
        for _ in range(episodes):
            rewards = episode_collector.collect_episode(max_episode_steps, epsilon=0., render=render)[2]
            episode_len += len(rewards)
            reward += rewards[-1]

        #TODO: ask tom: is loss need to be taken from the "collect_episode"-plaing or from buffer
        ####loss
        batch_size = self.config['model']['batch_size']
        current_state, action, reward, next_state, is_terminal = zip(*self.replay_buffer.sample_batch(batch_size))
        next_q_values = self.q_target_model.predict(np.array(next_state))
        max_next_q_value = np.max(next_q_values, axis=-1)
        target_labels = np.array(reward) + (1. - np.array(is_terminal)) * max_next_q_value
        one_hot_actions = self._one_hot_action(action)
        target_and_actions = np.concatenate((target_labels[:, None], one_hot_actions), axis=1)
        loss = self.q_model.test_on_batch(np.array(current_state), target_and_actions)
        ####end of loss

        episode_len = episode_len / episodes
        reward = reward / episodes
        summaries_collector.write_summaries('test', self.tests,reward, episode_len,loss)
        env.close()
        #print('TEST collected rewards: {}'.format(avg_reward))
        return reward