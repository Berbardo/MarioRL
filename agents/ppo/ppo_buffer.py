import numpy as np
import torch
import scipy.signal

class PPOBuffer:

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.length = 0

    def update(self, states, actions, log_probs, rewards, next_states, dones):
        self.states.append(states)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)
        self.length += 1

    def sample(self):
        states = self.states
        actions = self.actions
        log_probs = self.log_probs
        rewards = self.rewards
        next_states = self.next_states
        dones = self.dones
        self.reset()
        return states, actions, log_probs, rewards, next_states, dones