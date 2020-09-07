import numpy as np

from episode_collector import EpisodeCollector
from trajectory_replay_buffer import TrajectoryReplayBuffer
from replay_buffer import ReplayBuffer
from matplotlib import pyplot as plt
import seaborn as sns

class QLearning:
    def __init__(self, config, gym_wrapper,trajectory_ratio=0):
        self.config = config
        self.gym_wrapper = gym_wrapper
        env = self.gym_wrapper.get_env()

        self.map_size=int(np.sqrt(env.observation_space.n))
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        buf_size = self.config['model']['replay_buffer_size']
        #self.replay_buffer = TrajectoryReplayBuffer(buf_size,1) if trajectory_ratio==1 else ReplayBuffer(buf_size)
        self.replay_buffer = TrajectoryReplayBuffer(buf_size,trajectory_ratio)
        self.tests=0

    def train(self,summaries_collector):
        env = self.gym_wrapper.get_env()
        completion_reward = self.config['general']['completion_reward']
        epsilon = self.config['model']['epsilon']
        episode_collector = EpisodeCollector(self.q_table, env, self.gym_wrapper.get_num_actions())
        reward = 0
        episodes_len = 0
        loss=0
        for cycle in range(self.config['general']['cycles']):
            if not cycle%100:
                print('cycle {} epsilon {}'.format(cycle, epsilon))
            epsilon, cycle_reward,cycle_episode_len,cycle_loss = self._train_cycle(episode_collector, epsilon)
            reward +=cycle_reward
            episodes_len += cycle_episode_len
            loss += cycle_loss
            #save to csv
            if not cycle % 100:
                loss = loss / 100
                reward = reward / 100
                episodes_len = episodes_len / 100
                summaries_collector.write_summaries('train',cycle, reward, episodes_len,loss)
                reward = 0
                episodes_len = 0
                loss = 0

            #read once in 1000 episodes and plot into an image
            if (cycle%1000 ==0):
                summaries_collector.read_summaries('train')
                summaries_collector.read_summaries('test')

            if (cycle + 1) % self.config['general']['test_frequency'] == 0 or (completion_reward is not None and (cycle_reward > completion_reward)):
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
        #if (avg_rewards > 0.5 ):
            #print('collected rewards: {}'.format(avg_rewards))
        epsilon *= self.config['model']['decrease_epsilon']
        epsilon = max(epsilon, self.config['model']['min_epsilon'])
        steps = self.config['model']['train_steps_per_cycle']
        # train steps
        cycle_loss = 0
        for _ in range(steps):
            cycle_loss+= self._train_step()
        episodes = self.config['general']['episodes_per_training_cycle']
        return epsilon, avg_rewards, cycle_len/episodes, cycle_loss/steps

    def _train_step(self):
        gamma = self.config['general']['gamma']
        lr = self.config['policy_network']['learn_rate']
        current_state, action, reward, next_state, is_terminal = zip(*self.replay_buffer.sample_batch(1))
        next_q_values = self.q_table[next_state[0]]
        max_next_q_value = np.max(next_q_values, axis=-1)
        #bellman equation
        q_label = reward[0] + gamma*(1. - is_terminal[0]) * max_next_q_value
        curr_q_label = self.q_table[current_state[0], action[0]]
        self.q_table[np.array(current_state[0]),action[0]] =curr_q_label*(1-lr) + lr* q_label
        step_loss = (np.square(q_label - curr_q_label))
        return step_loss

    def test(self,summaries_collector, render=False, episodes=None):
        self.tests+=1
        env = self.gym_wrapper.get_env()
        episode_collector = EpisodeCollector(self.q_table, env, self.gym_wrapper.get_num_actions())
        max_episode_steps = self.config['general']['max_test_episode_steps']
        if episodes is None:
            episodes = self.config['general']['episodes_per_test']
        episode_len = 0
        reward = 0
        for episode in range(episodes):
            rewards = episode_collector.collect_episode(max_episode_steps, epsilon=0., render=render)[2]
            episode_len += len(rewards)
            reward += rewards[-1]
            #self.calc_loss(states, actions, rewards, is_terminal, self.config['general']['gamma'])

        episode_len = episode_len / episodes
        reward = reward / episodes

        summaries_collector.write_summaries('test', self.tests,reward, episode_len)

        env.close()
        #print('TEST collected rewards: {}'.format(avg_reward))
        return reward



    #NOT TO RUN YET
    def calc_loss(self,states, actions, rewards, is_terminal, gamma):
        loss=0
        for i in range(len(rewards)):
            q_label = np.array(rewards[i]) + gamma * (1. - is_terminal[i]) * np.max(self.q_table[states[i+1]])
            loss += np.square(q_label -self.q_table[states[i]][actions[i]])
        return loss



    def loss_evaluation(self):
        gamma = self.config['general']['gamma']
        rewards = np.zeros((self.map_size, self.map_size))
        rewards[self.map_size -1, self.map_size- 1] = 1
        terminal = np.zeros((self.map_size, self.map_size))
        if ( self.map_size==4):

            terminal[1, 1] = terminal[1, 3] = terminal[2, 3] = terminal[3, 0] = terminal[3,3]=1

            next_state_vec = np.zeros(self.map_size**2)
            for i in range(len(next_state_vec)):
                next_state_vec[i]= np.argmax(self.q_table[i])

            next_state_q_vec = np.zeros(self.map_size**2)
            for i in range(len(next_state_q_vec)):
                next_state_q_vec[i] = np.max(self.q_table[int(next_state_vec[i])])
            next_state_q = next_state_vec.reshape(self.map_size, self.map_size)

            losses = rewards + gamma*(1-terminal)*next_state_q

            avg_loss = [np.mean((losses[1, 0], losses[0, 1])), np.mean((losses[0, 2], losses[2, 0])),
                        np.mean((losses[0, 3], losses[1, 2], losses[2, 1])), np.mean((losses[3, 1], losses[2, 2])),
                        losses[3, 2], losses[3, 3]]
        else:

            terminal[7, 7] = terminal[2, 3] = terminal[3, 5] = terminal[4, 3] = terminal[5, 1]= terminal[5, 2]=1
            terminal[5, 6] = terminal[6, 1] =  terminal[6, 4] =  terminal[6, 6] =terminal[7, 3]  = 1

            next_state_vec = np.zeros(self.map_size ** 2)
            for i in range(len(next_state_vec)):
                next_state_vec[i] = np.argmax(self.q_table[i])

            next_state_q_vec = np.zeros(self.map_size ** 2)
            for i in range(len(next_state_q_vec)):
                next_state_q_vec[i] = np.max(self.q_table[int(next_state_vec[i])])
            next_state_q = next_state_vec.reshape(self.map_size, self.map_size)

            losses = rewards + gamma * (1 - terminal) * next_state_q

            avg_loss = [np.mean((losses[1, 0], losses[0, 1])), np.mean((losses[0, 2], losses[2, 0],losses[1,1])),
                        np.mean((losses[0, 3], losses[1, 2], losses[2, 1], losses [3,0])),
                        np.mean((losses[4,0],losses[3, 1], losses[2, 2],losses[1,3],losses[0,4])),
                        np.mean((losses[4, 0], losses[3, 1], losses[2, 2], losses[1, 3], losses[0, 4])),
                        np.mean((losses[5, 0], losses[4, 1], losses[3, 2], losses[1, 4], losses[0, 5])),
                        np.mean((losses[6, 0], losses[4, 2], losses[3, 3], losses[2, 4], losses[1, 5],losses[0,6])),
                        np.mean((losses[7, 0], losses[4, 3], losses[3, 4], losses[2, 5], losses[1, 6], losses[0, 7])),
                        np.mean((losses[7, 1], losses[6, 2], losses[5, 3], losses[4, 4], losses[2, 6], losses[1, 7])),
                        np.mean((losses[7, 2], losses[6, 3], losses[5, 4], losses[4, 5], losses[3, 6],losses[2, 7])),
                        np.mean((losses[5, 5], losses[4, 6], losses[3, 7])),
                        np.mean((losses[7, 4], losses[6, 5], losses[4, 7])),
                        np.mean((losses[7, 5], losses[5, 7])),
                        np.mean((losses[7, 6], losses[6, 7])),
                        ]
        """
        plt.title(" losses by distance")
        color = 'g' if self.trajectory else 'r'
        plt.plot(np.linspace(1, 6, 6), avg_loss, color=color)
        # plt.legend()
        plt.savefig('Loss by distance.png')
        """
        return avg_loss


