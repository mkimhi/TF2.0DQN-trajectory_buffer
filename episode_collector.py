import numpy as np

class EpisodeCollector:
    def __init__(self, q_learner, env, num_actions,is_deep=False):
        self.q_learner = q_learner
        self.env = env
        self.num_actions = num_actions
        self.is_deep=is_deep

    def _get_action(self, state, epsilon):
        if epsilon > 0. and np.random.uniform() < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            state_ = np.expand_dims(np.array(state), 0)
            action_logits = self.q_learner.predict(state_)[0] if (self.is_deep) else self.q_learner[state,:]
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