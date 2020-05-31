import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.dqn.prioritized_replay import PrioritizedReplayBuffer

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()
        self.out_dim = out_dim

        self.convs = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())

        self.advantage1 = nn.Linear(7 * 7 * 64, 512)
        self.advantage2 = nn.Linear(512, out_dim)

        self.value1 = nn.Linear(7 * 7 * 64, 512)
        self.value2 = nn.Linear(512, 1)

    def forward(self, state):
        conv = self.convs(state)
        flat = conv.reshape(-1, 7 * 7 * 64)
        adv_hid = F.relu(self.advantage1(flat))
        val_hid = F.relu(self.value1(flat))

        advantages = self.advantage2(adv_hid)
        values = self.value2(val_hid)

        q = values + (advantages - advantages.mean())

        return q

class DQN:
    def __init__(self, observation_space, action_space, lr=2.5e-4, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.beta = 0.6
        self.memory = PrioritizedReplayBuffer(20000, 0.6)
        self.action_space = action_space

        self.epsilon = 0.1
        self.epsilon_decay = 0.9998
        self.min_epsilon = 0.01

        self.update_count = 0
        self.dqn = Network(observation_space.shape[0], action_space.n).to(self.device)
        self.dqn_target = Network(observation_space.shape[0], action_space.n).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer  = optim.Adam(self.dqn.parameters(), lr=lr)

    def act(self, state, evaluate=False):
        if not evaluate:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)

            if np.random.random() < self.epsilon:
                action = [self.action_space.sample() for i in range(len(state))]
                return action

        state = torch.FloatTensor(state).to(self.device)
        action = self.dqn.forward(state).argmax(dim=-1)
        action = action.cpu().detach().numpy()

        return action

    def remember(self, states, actions, rewards, new_states, dones):
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], new_states[i], dones[i])

    def train(self, batch_size=32, epochs=1):
        if 10000 > len(self.memory._storage):
            return
        
        for epoch in range(epochs):
            self.update_count +=1

            if self.update_count % 2:
                return

            self.beta = self.beta + self.update_count/1000000 * (1.0 - self.beta)

            (states, actions, rewards, next_states, dones, weights, batch_indexes) = self.memory.sample(batch_size, self.beta)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
            weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.device)

            q = self.dqn.forward(states).gather(-1, actions.long())
            a2 = self.dqn.forward(next_states).argmax(dim=-1, keepdim=True)
            q2 = self.dqn_target.forward(next_states).gather(-1, a2).detach()

            target = (rewards + (1 - dones) * self.gamma * q2).to(self.device)

            td_error = F.smooth_l1_loss(q, target, reduction="none")
            loss = torch.mean(td_error * weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_target()

            priorities = td_error.detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(batch_indexes, priorities)

    def update_target(self):
        for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def hard_update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def save_model(self, path):
        torch.save(self.dqn.state_dict(), path)
    
    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.hard_update_target()
