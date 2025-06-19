import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

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

                state = torch.from_numpy(observation).view(1, -1)

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
    
    def playback(self):
        episode_reward = []
        frames = []

        observation, info = self.env.reset()
        episode_over = False
        while not episode_over:
            frames.append(self.env.render())

            with torch.no_grad():
                probs = self.policy(torch.from_numpy(observation).view(1, -1))
                action = torch.argmax(probs, dim=1).item()

            observation, reward, terminated, truncated, info = self.env.step(action)
            episode_reward.append(reward)

            episode_over = terminated or truncated

        print(f"Playback Return: {sum(episode_reward)}")

        self.env.close()

        return frames, episode_reward

    def eval(self, episodes):
        rewards = []
        for _ in range(episodes):
            observation, info = self.env.reset()
            episode_reward = 0
            episode_over = False
            while not episode_over:

                with torch.no_grad():
                    probs = self.policy(torch.from_numpy(observation).view(1, -1))
                    action = torch.argmax(probs, dim=1).item()

                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                episode_over = terminated or truncated

            rewards.append(episode_reward)

        print(f"Average Return over {episodes} episodes: {sum(rewards) / episodes}")

        self.env.close()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)