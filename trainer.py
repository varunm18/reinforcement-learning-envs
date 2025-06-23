import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import animation

from collections import defaultdict, deque

# DDQN (Double DQN)
class DDQN:
    def __init__(self, env, q_net_class, buffer_size, batch_size,
                 gamma, lr, tau, epsilon_start, epsilon_end, 
                 epsilon_decay, target_update_interval):

        self.env = env

        self.q_net = q_net_class()
        self.target_net = q_net_class()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_interval = target_update_interval
        self.learn_step = 0

        self.losses = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.tensor(state).float().view(1, -1)
        with torch.no_grad():
            q_values = self.q_net(state)
        return torch.argmax(q_values).item()

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        # convert (s, a, r, s, d) list into tuples of all s, a, r, s, d
        state, action, reward, next_state, done = zip(*batch)
        return (torch.tensor(state).float(),               # batch x state size
                torch.tensor(action).view(-1, 1),          # batch x 1
                torch.tensor(reward).float().view(-1, 1),  # batch x 1
                torch.tensor(next_state).float(),          # batch x state size
                torch.tensor(done).float().view(-1, 1))    # batch x 1

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values = self.q_net(states).gather(1, actions)

        # DDQN: use main network to choose actions, target network to evaluate
        next_actions = self.q_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions)

        target_q = rewards + (1 - dones) * self.gamma * next_q_values
        
        # detach target_q from loss computation 
        loss = self.criterion(q_values, target_q.detach())
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.learn_step += 1

    def train(self, episodes, stats_interval):
        rewards = []

        for e in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            episode_over = False
            while not episode_over:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_over = terminated or truncated

                self.buffer.append((state, action, reward, next_state, episode_over))
                self.train_step()

                state = next_state
                episode_reward += reward
                if episode_over:
                    break

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            rewards.append(episode_reward)

            if (e + 1) % stats_interval == 0:
                print(f"Episodes {(e - stats_interval + 1, e)}:", end=" ")
                print(f"Avg Reward- {np.mean(rewards[-stats_interval:])}", end=" ")
                print(f"Epsilon- {self.epsilon}")

        return rewards, self.losses

    def playback(self, fps=30, text_color="black"):
        episode_reward = []
        frames = []

        observation, _ = self.env.reset()
        episode_over = False
        while not episode_over:
            frames.append(self.env.render())

            with torch.no_grad():
                q_vals = self.q_net(torch.tensor(observation).float().view(1, -1))
                action = torch.argmax(q_vals, dim=1).item()

            observation, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward.append(reward)

            episode_over = terminated or truncated

        print(f"Playback Return: {sum(episode_reward)}")

        self.env.close()

        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames[0])

        reward_text = ax.text(10, 20, "", color=text_color, fontsize=12, weight='bold')

        def animate(i):
            im.set_array(frames[i])
            r = sum(episode_reward[:i+1])
            reward_text.set_text(f"Return: {r:.2f}")
            return [im, reward_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps)
        plt.close()
        return anim.to_jshtml()

    def eval(self, episodes):
        rewards = []
        for _ in range(episodes):
            observation, _ = self.env.reset()
            episode_reward = 0
            episode_over = False
            while not episode_over:

                with torch.no_grad():
                    q_vals = self.q_net(torch.tensor(observation).float().view(1, -1))
                    action = torch.argmax(q_vals, dim=1).item()

                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                episode_over = terminated or truncated

            rewards.append(episode_reward)

        print(f"Average Return over {episodes} episodes: {sum(rewards) / episodes}")

        self.env.close()

# Q-Learning with Temporal Difference (TD)
class QLearn():
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
        self.env = env
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration rate
        self.epsilon_decay = epsilon_decay # exploration decay rate
        self.min_epsilon = min_epsilon # minimum exploration rate

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def train(self, episodes, stats_interval):
        rewards = []
        max_reward = 0

        for e in range(episodes):
            observation, _ = self.env.reset()

            episode_reward = 0

            episode_over = False
            while not episode_over:

                if random.uniform(0, 1) <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[observation])

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                last_reward = reward
                episode_reward += reward

                old_value = self.q_table[observation][action]
                next_max = np.max(self.q_table[next_observation])

                self.q_table[observation][action] = ((1 - self.alpha) * old_value) + (self.alpha * (reward + self.gamma * next_max))

                observation = next_observation

                episode_over = terminated or truncated
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            
            if terminated:
                episode_reward = last_reward
            
            rewards.append(episode_reward)

            if episode_reward > max_reward:
                max_reward = episode_reward
                print(f"New max of {max_reward} in episode {e + 1}")

            if (e + 1) % stats_interval == 0:
                print(f"Episodes {(e - stats_interval + 1, e)}:", end=" ")
                print(f"Avg Reward- {np.mean(rewards[-stats_interval:])}", end=" ")
                print(f"Epsilon- {self.epsilon}")
                
        self.env.close()

        return rewards

    def playback(self, fps=30):
        episode_reward = []
        frames = []

        observation, _ = self.env.reset()
        episode_over = False
        while not episode_over:
            frames.append(self.env.render())

            action = np.argmax(self.q_table[observation])

            observation, reward, terminated, truncated, _ = self.env.step(action)

            episode_reward.append(reward)

            episode_over = terminated or truncated

        print(f"Playback Return: {episode_reward[-1] if terminated else sum(episode_reward)}")

        self.env.close()

        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames[0])

        reward_text = ax.text(10, 20, "", color="black", fontsize=12, weight='bold')

        def animate(i):
            im.set_array(frames[i])
            r = sum(episode_reward[:i+1])
            if terminated and i == len(frames) - 1:
                r = episode_reward[-1]
            reward_text.set_text(f"Return: {r:.2f}")
            return [im, reward_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps)
        plt.close()
        return anim.to_jshtml()

    def eval(self, episodes):
        rewards = []
        for _ in range(episodes):
            observation, _ = self.env.reset()
            episode_reward = 0
            episode_over = False
            while not episode_over:

                action = np.argmax(self.q_table[observation])

                observation, reward, terminated, truncated, _ = self.env.step(action)
                last_reward = reward
                episode_reward += reward

                episode_over = terminated or truncated

            if terminated:
                episode_reward = last_reward

            rewards.append(episode_reward)

        print(f"Average Return over {episodes} episodes: {sum(rewards) / episodes}")

        self.env.close()


# REINFORCE for discrete action space
class Reinforce():
    def __init__(self, env, policy, optimizer, discount):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.discount = discount
    
    def train(self, episodes, stats_interval):
        losses = []
        rewards = []
        max_reward = 0

        for e in range(episodes):
            observation, _ = self.env.reset()

            log_probs = []
            episode_rewards = []
            total_episode_reward = 0

            episode_over = False
            while not episode_over:

                state = torch.tensor(observation).float().view(1, -1)

                # action = env.action_space.sample()  # agent policy that uses the observation and info
                probs = self.policy(state)
                action = torch.multinomial(probs, num_samples=1).item()
                observation, reward, terminated, truncated, _ = self.env.step(action)

                log_probs.append(torch.log(probs[0, action]))
                episode_rewards.append(reward)
                total_episode_reward += reward

                episode_over = terminated or truncated
            
            rewards.append(total_episode_reward)
            if total_episode_reward > max_reward:
                max_reward = total_episode_reward
                print(f"New max of {max_reward} in episode {e + 1}")

            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + self.discount * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            if returns.size(dim=-1) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            loss = 0
            self.optimizer.zero_grad()
            for log_prob, R in zip(log_probs, returns):
                loss += -log_prob * R
                
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

            if (e + 1) % stats_interval == 0:
                print(f"Episodes {(e - stats_interval + 1, e)}:", end=" ")
                print(f"Avg Loss- {sum(losses[e-stats_interval+1:e+1])/stats_interval}", end=" ")
                print(f"Avg Reward- {sum(rewards[e-stats_interval+1:e+1])/stats_interval}")
                
        self.env.close()

        return rewards, losses
    
    def playback(self, fps=30, text_color="black"):
        episode_reward = []
        frames = []

        observation, _ = self.env.reset()
        episode_over = False
        while not episode_over:
            frames.append(self.env.render())

            with torch.no_grad():
                probs = self.policy(torch.tensor(observation).float().view(1, -1))
                action = torch.argmax(probs, dim=1).item()

            observation, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward.append(reward)

            episode_over = terminated or truncated

        print(f"Playback Return: {sum(episode_reward)}")

        self.env.close()

        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames[0])

        reward_text = ax.text(10, 20, "", color=text_color, fontsize=12, weight='bold')

        def animate(i):
            im.set_array(frames[i])
            r = sum(episode_reward[:i+1])
            reward_text.set_text(f"Return: {r:.2f}")
            return [im, reward_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps)
        plt.close()
        return anim.to_jshtml()

    def eval(self, episodes):
        rewards = []
        for _ in range(episodes):
            observation, _ = self.env.reset()
            episode_reward = 0
            episode_over = False
            while not episode_over:

                with torch.no_grad():
                    probs = self.policy(torch.tensor(observation).float().view(1, -1))
                    action = torch.argmax(probs, dim=1).item()

                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                episode_over = terminated or truncated

            rewards.append(episode_reward)

        print(f"Average Return over {episodes} episodes: {sum(rewards) / episodes}")

        self.env.close()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)