from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add_episode(self, states, actions, rewards, is_terminal):
        # make sure all the sizes match, there is one more state then everything else
        assert len(states) == len(actions) + 1
        assert len(actions) == len(rewards)
        assert len(rewards) == len(is_terminal)
        # assert that the only terminal state is the last one
        assert not any(is_terminal[:-1])
        # partition the states to "current" and "next"
        current_sates = states[:-1]
        next_states = states[1:]
        # zip and add
        transitions = zip(current_sates, actions, rewards, next_states, is_terminal)
        for t in transitions:
            self.add_transition(t)

    def add_transition(self, transition):
        if self.count() >= self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def count(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch_size = min([batch_size, self.count()])
        batch = random.sample(self.buffer, batch_size)
        return batch
