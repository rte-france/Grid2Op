from collections import deque
import random
import numpy as np

class ExperienceBuffer:

    def __init__(self, buffer_size, batch_size, trace_length):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.trace_length = trace_length
        self.buffer = [[]]
        self.current_episode = 0

    def add(self, s, a, r, d, s2, e):
        experience = np.reshape(np.array([s, a, r, d, s2]), [1,5])

        # New episode
        if self.current_episode < e:
            self.current_episode = e
            
            if self.size_episode() < self.trace_length:
                # Last episode was too short, reuse buffer
                self.buffer[-1] = []
            else:
                # Create new episode entry
                self.buffer.append([])
                # Forget oldest episode to respect size limits
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)

        # Save experience
        self.buffer[-1].append(experience)

    def size(self):
        return len(self.buffer)

    def size_episode(self, episode_index = -1):
        return len(self.buffer[episode_index])

    def can_sample(self):
        if self.size() < self.batch_size:
            return False
        if self.size_episode() < self.trace_length:
            return False
        return True

    def sample(self):
        samples = []
        # Get random episodes
        sampled_episodes = random.sample(self.buffer, self.batch_size)
        # Get random trace in each episode
        for episode in sampled_episodes:
            max_trace_start = len(episode) - self.trace_length + 1
            trace_start = np.random.randint(0, max_trace_start)
            trace_end = trace_start + self.trace_length
            sample = episode[trace_start:trace_end]
            samples.append(sample)

        samples = np.array(samples)
        return np.reshape(samples, [self.batch_size * self.trace_length, 5])

    def clear(self):
        self.buffer = [[]]
        self.current_episode = 0
