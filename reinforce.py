import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class REINFORCE:
    def __init__(self, lr=3e-4, gamma=0.99):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.baseline = None

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def calculate_returns(self, rewards):
        returns = []
        G  = 0
        for r in reversed(rewards):
            G = r + self.gamma*G
            returns.insert(0, G)
        return returns

    def update_policy(self, log_probs, returns):
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        
        # use an advantage over baseline (mean weighted) reward to boost signal
        if self.baseline is None:
            self.baseline = returns.mean()
        else:
            self.baseline = 0.95 * self.baseline + 0.05 * returns.mean()

        advantages = returns - self.baseline
        loss = -(log_probs * advantages).mean()  # mean instead of sum. with mean, each episode contributes roughly the same scale of gradient regardless of length.

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def rollout(self, env):
        state = env.reset()
        rewards = []
        log_probs = []
        done = False

        while not done:
            action, log_prob = self.select_action(state) 
            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        returns = self.calculate_returns(rewards)

        loss = self.update_policy(log_probs, returns)

        return sum(rewards), loss

def train(num_episodes=2000):
    env = gym.make("LunarLander-v2")
    agent = REINFORCE()

    episode_rewards = []

    for i in range(num_episodes):
        return_total, loss = agent.rollout(env)
        episode_rewards.append(return_total)

        if (i+1) % 100 == 0: # print every 100 episodes
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {i+1}, avg reward: {avg_reward:.2f}, loss: {loss:.2f}")
    env.close()
    return agent, episode_rewards

if __name__ == '__main__':
    print('training reinforce')
    agents, rewards = train()
    print(f"\nFinal 100-episode average: {np.mean(rewards[-100:]):.2f}")

