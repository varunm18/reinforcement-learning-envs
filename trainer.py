import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import animation

from collections import defaultdict

# Q-Learning
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
                print(f"Avg Reward- {sum(rewards[e-stats_interval+1:e+1])/stats_interval}")
                
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