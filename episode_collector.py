import numpy as np


class EpisodeCollector:
    def __init__(self, q_network, env, num_actions):
        self.q_network = q_network
        self.env = env
        self.num_actions = num_actions

    def _get_action(self, state, epsilon):
        if epsilon > 0. and np.random.uniform() < epsilon:
            # act randomly
            action = np.random.choice(self.num_actions)
        else:
            # act according to model
            state_ = np.expand_dims(np.array(state), 0)
            action_logits = self.q_network.predict(state_)[0]
            action = np.argmax(action_logits, axis=-1)
        return action

    def collect_episode(self, max_steps, epsilon=0., render=False):
        states, actions, rewards, is_terminal = [self.env.reset()], [], [], []
        for step in range(max_steps):
            if render:
                self.env.render()
            current_state = states[-1]
            action = self._get_action(current_state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            is_terminal.append(done)
            if done:
                if render:
                    self.env.render()
                break
        return states, actions, rewards, is_terminal
