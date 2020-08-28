import numpy as np

from episode_collector import EpisodeCollector
from trajectory_replay_buffer import TrajectoryReplayBuffer
from replay_buffer import ReplayBuffer


class QLearning:
    def __init__(self, config, gym_wrapper,trajectory=True):
        self.config = config
        self.gym_wrapper = gym_wrapper
        env = self.gym_wrapper.get_env()
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        buf_size = self.config['model']['replay_buffer_size']
        self.replay_buffer = TrajectoryReplayBuffer(buf_size) if trajectory else ReplayBuffer(buf_size)
        self.tests=0
        #self.loss=[]

    def train(self,summaries_collector):
        env = self.gym_wrapper.get_env()
        completion_reward = self.config['general']['completion_reward']
        epsilon = self.config['model']['epsilon']
        episode_collector = EpisodeCollector(self.q_table, env, self.gym_wrapper.get_num_actions())
        accumulated_reward = 0
        accumulated_episodes_len = 0
        accumulated_loss=0
        for cycle in range(self.config['general']['cycles']):
            #print('cycle {} epsilon {}'.format(cycle, epsilon))
            #TODO: try both with cycle%100 and direct from train_cycle output and check what's better
            epsilon, avg_reward_per_cycle,avg_episode_len_per_cycle,avg_loss_per_cycle = self._train_cycle(episode_collector, epsilon)
            accumulated_reward +=avg_reward_per_cycle
            accumulated_episodes_len += avg_episode_len_per_cycle
            accumulated_loss += avg_loss_per_cycle
            #save to csv
            if (cycle%100 ==0):
                reward = accumulated_reward/100
                episodes_len = accumulated_episodes_len/100
                loss = accumulated_loss/100
                summaries_collector.write_summaries('train',cycle, reward, episodes_len,loss)
                accumulated_reward = 0
                accumulated_episodes_len = 0
                accumulated_loss = 0

            #read once in 1000 episodes and plot into an image
            if (cycle%1000 ==0):
                summaries_collector.read_summaries('train')
                summaries_collector.read_summaries('test')

            if (cycle + 1) % self.config['general']['test_frequency'] == 0 or (completion_reward is not None and avg_reward_per_cycle > completion_reward):
                test_avg_reward = self.test(summaries_collector)
                if completion_reward is not None and test_avg_reward > completion_reward:
                    print('TEST avg reward {} > required reward {}... stopping training'.format(test_avg_reward, completion_reward))
                    break
        env.close()

    def _train_cycle(self, episode_collector, epsilon):
        cycle_len = 0
        # collect data
        max_episode_steps = self.config['general']['max_train_episode_steps']
        rewards_per_episode = []
        for _ in range(self.config['general']['episodes_per_training_cycle']):
            states, actions, rewards, is_terminal_flags = episode_collector.collect_episode(
                max_episode_steps, epsilon=epsilon)
            self.replay_buffer.add_episode(states, actions, rewards, is_terminal_flags)
            #reward_by_len = np.mean(rewards)
            rewards_per_episode.append(sum(rewards))
            cycle_len += len(actions)
        avg_rewards = np.mean(rewards_per_episode)
        if (avg_rewards > 0.5 ):
            print('collected rewards: {}'.format(avg_rewards))
        epsilon *= self.config['model']['decrease_epsilon']
        epsilon = max(epsilon, self.config['model']['min_epsilon'])

        #todo: try to clean the code!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # train steps
        cycle_loss = 0
        for _ in range(self.config['model']['train_steps_per_cycle']):
            cycle_loss+= self._train_step()
        eps_per_cyc = self.config['general']['episodes_per_training_cycle']
        return epsilon, avg_rewards, cycle_len/eps_per_cyc, cycle_loss/self.config['model']['train_steps_per_cycle']

    def _train_step(self):
        step_loss=0
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
            step_loss+=np.square(q_label - curr_q_label)
            print("step loss: ",step_loss)
            return step_loss



    def test(self,summaries_collector, render=False, episodes=None):
        self.tests+=1

        env = self.gym_wrapper.get_env()
        episode_collector = EpisodeCollector(self.q_table, env, self.gym_wrapper.get_num_actions())
        max_episode_steps = self.config['general']['max_test_episode_steps']
        if episodes is None:
            episodes = self.config['general']['episodes_per_test']
        avg_episode_len = 0
        avg_reward = 0
        for episode in range(episodes):
            rewards = episode_collector.collect_episode(max_episode_steps, epsilon=0., render=render)[2]
            avg_episode_len += len(rewards)
            avg_reward += rewards[-1]
            calc_loss(states, actions, rewards, is_terminal, self.config['general']['gamma'])

        avg_episode_len = avg_episode_len / episodes
        avg_reward = avg_reward / episodes

        summaries_collector.write_summaries('test', self.tests,avg_reward, avg_episode_len)

        env.close()
        #print(self.q_table)
        #print('TEST collected rewards: {}'.format(avg_reward))
        return avg_reward

    #NOT TO RUN YET
    def calc_loss(self,states, actions, rewards, is_terminal, gamma):
        loss=0
        for i in range(len(rewards)):
            q_label = np.array(rewards[i]) + gamma * (1. - is_terminal[i]) * np.max(self.q_table[states[i+1]])
            loss += np.square(q_label -self.q_table[states[i]][actions[i]])
        return loss



