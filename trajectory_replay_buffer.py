from collections import deque
import numpy as np
import random



"""
class episode:
    def __init__(self):
        self.deque=deque()
        self.a=1
        self.b=0.5
        self.len=0

    def add_transition(self,t):
        self.deque.append(t)
        self.len +=1

    def sample_episode(self):
        index= int(np.random.beta(self.a,self.b)*self.len)
        self.len-=1
        self.change_a_b()
        return(self.deque[index+1])

    def change_a_b(self):
        if(self.b != 1):
            self.b+=0.1
        elif(self.a == 0.5):
            self.a=1
            self.b=0.5
        else:
            self.a-=0.1
"""

class TrajectoryReplayBuffer:
    def __init__(self, buffer_size,traj_ratio,min_traj_ratio,decrease_trajectory_ratio):
        self.buffer_size = buffer_size
        self.current_size = 0
        self.trajectory_buffer = deque()
        self.buffer = deque()
        self.traj_ratio = traj_ratio
        self.min_traj_ratio=min_traj_ratio
        self.decrease_trajectory_ratio=decrease_trajectory_ratio

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
        while self.current_size > self.buffer_size:
            popped_episode = self.trajectory_buffer.popleft()
            self.current_size -= len(popped_episode)
        self.trajectory_buffer.append(deque())
        self.current_size += len(rewards)
        # zip and add
        transitions = zip(current_sates, actions, rewards, next_states, is_terminal)
        for t in transitions:
            self.add_transition_trajectory(t)
            self.add_transition(t)

    def add_transition(self, transition):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def add_transition_trajectory(self, transition):
        self.trajectory_buffer[-1].append(transition)

    def sample_batch(self, batch_size):
        batch_size = min([batch_size, self.current_size,len(self.buffer)])
        if random.random() > self.traj_ratio:  # regular buffer
            batch = random.sample(self.buffer, batch_size)
            #print("------")
        else:
            updated_ratio = self.traj_ratio * self.decrease_trajectory_ratio
            self.traj_ratio = updated_ratio if (updated_ratio > self.min_traj_ratio) else self.traj_ratio
            #print(self.traj_ratio)

            indexes = np.random.randint(len(self.trajectory_buffer), size=batch_size)
            batch = []
            for i in indexes:
                batch.append(self.trajectory_buffer[i][-1])
                self.trajectory_buffer[i].rotate()

        return batch
