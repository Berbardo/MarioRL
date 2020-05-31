import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.ppo.ppo_buffer import PPOBuffer

class ActorCritic(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convs = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())

        self.lin = nn.Linear(7 * 7 * 64, 512)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))

        self.pi_logits = nn.Linear(512, action_shape)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))

        self.value = nn.Linear(512, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs):
        conv = self.convs(obs)
        flat = conv.reshape(-1, 7 * 7 * 64)

        h = F.relu(self.lin(flat))

        probs = F.softmax(self.pi_logits(h), dim=-1)
        probs = Categorical(probs)
        value = self.value(h)

        return probs, value

class PPO:
    def __init__(self, observation_space, action_space, lr=2.5e-4, steps=256, gamma=0.99, lam=0.95, entropy_coef=0.005, clip=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.clip = clip
        self.steps = steps

        self.memory = PPOBuffer()

        self.actorcritic = ActorCritic(observation_space.shape[0], action_space.n).to(self.device)
        self.actorcritic_optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr, eps=1e-6)
        self.target_actorcritic = ActorCritic(observation_space.shape[0], action_space.n).to(self.device)
        self.target_actorcritic.load_state_dict(self.actorcritic.state_dict())

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.target_actorcritic.forward(state)

        if evaluate:
            action = probs.probs.argmax(dim=-1)
        else:
            action = probs.sample()
        
        return action.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        state_torch = torch.FloatTensor(state).to(self.device)
        probs, _ = self.actorcritic.forward(state_torch)
        action_torch = torch.LongTensor(action).to(self.device)
        log_probs = probs.log_prob(action_torch)
        self.memory.update(state, action, log_probs, reward, new_state, done)

    def compute_gae(self, values, dones, rewards):
        returns = torch.zeros_like(rewards).to(self.device)
        advantages = torch.zeros_like(rewards).to(self.device)
        deltas = torch.zeros_like(rewards).to(self.device)

        returns[-1] = rewards[-1] + self.gamma * (1 - dones[-1]) * rewards[-1]
        advantages[-1] = returns[-1] - values[-1]

        for i in reversed(range(len(rewards) - 1)):
            delta = rewards[i] + self.gamma * (1 - dones[i]) * values[i+1] - values[i]
            advantages[i] = delta + self.gamma * self.lam * (1 - dones[i]) * advantages[i + 1]
            returns[i] = advantages[i] + values[i]

        return returns, (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    def compute_loss(self, states, actions, logp, advantages, returns):
        new_probs, v = self.actorcritic.forward(states)
        
        new_logprobs = new_probs.log_prob(actions)
        entropy = new_probs.entropy().mean()
        ratios = torch.exp(new_logprobs.unsqueeze(-1) - logp.unsqueeze(-1).detach())

        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages.detach()

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * F.mse_loss(v, returns.detach())
        entropy_loss = - self.entropy_coef * entropy

        return policy_loss, value_loss, entropy_loss

    def train(self, epochs=8):
        if self.memory.length < self.steps:
            return

        states, actions, log_probs, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device).detach()

        returns = []
        advantages = []

        for env_num in range(8):
            _, v = self.actorcritic.forward(states[:,env_num])
            rt, adv = self.compute_gae(v, dones[:,env_num], rewards[:,env_num])
            returns.append(rt)
            advantages.append(adv)

        for _ in range(epochs):
            for env_num in range(8):
                self.actorcritic_optimizer.zero_grad()

                policy_loss, value_loss, entropy_loss = self.compute_loss(states[:,env_num], actions[:,env_num], 
                                                            log_probs[:,env_num], advantages[env_num], returns[env_num])            
                total_loss = policy_loss + value_loss + entropy_loss

                total_loss.backward()
                self.actorcritic_optimizer.step()

        self.target_actorcritic.load_state_dict(self.actorcritic.state_dict())

    def save_model(self, path):
        torch.save(self.actorcritic.state_dict(), path)

    def load_model(self, path):
        self.actorcritic.load_state_dict(torch.load(path))
        self.target_actorcritic.load_state_dict(self.actorcritic.state_dict())