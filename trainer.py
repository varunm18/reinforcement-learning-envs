import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib import animation

from collections import defaultdict, deque

from abc import ABC, abstractmethod

def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

class Agent(ABC):
    def __init__(self, env):
        self.env = env
        self.episodes_trained = 0

    @abstractmethod
    def select_action(self, observation):
        """
        Select an action based on the current observation.
        """
        pass

    def play_episode(self, track_frames=False, grad=False):
        """
        Play a single episode in the environment.
        """
        observation, _ = self.env.reset()

        episode_reward = []
        if track_frames:
            frames = []

        episode_over = False
        while not episode_over:
            if track_frames:
                frames.append(self.env.render())

            with torch.set_grad_enabled(grad):
                action = self.select_action(observation)

            observation, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward.append(reward)

            episode_over = terminated or truncated

        return episode_reward, terminated, truncated, (frames if track_frames else None)
    
    @abstractmethod
    def train(self, episodes, stats_interval):
        """
        Train the agent for a number of episodes.
        """
        pass
    
    def playback(self, fps=30, text_color="black", show_last_reward=False):
        """
        Play back the agent's performance in the environment.
        """
        episode_reward, terminated, truncated, frames = self.play_episode(track_frames=True)
        print(f"Playback Return: {episode_reward[-1] if show_last_reward and terminated else sum(episode_reward)}")

        self.env.close()

        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames[0])

        reward_text = ax.text(10, 20, "", color=text_color, fontsize=12, weight='bold')

        def animate(i):
            im.set_array(frames[i])

            if show_last_reward and terminated and i == len(frames) - 1:
                r = episode_reward[-1]
            else:
                r = sum(episode_reward[:i+1])

            reward_text.set_text(f"Return: {r:.2f}")
            return [im, reward_text]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1000/fps)
        plt.close()
        return anim.to_jshtml() 
    
    def eval(self, episodes, save_last_reward=False):
        """ 
        Evaluate the agent over a number of episodes.
        If save_last is True, only the last reward of 
        each episode is saved if terminated.
        """
        rewards = []
        for _ in range(episodes):
            episode_reward, terminated, truncated, _ = self.play_episode()

            if save_last_reward and terminated:
                rewards.append(episode_reward[-1])
            else:
                rewards.append(sum(episode_reward))

        print(f"Average Return over {episodes} episodes: {np.mean(rewards)}")

        self.env.close()

    @abstractmethod
    def save(self, path):
        """
        Save the agent's model to a file.
        """
        pass

    @abstractmethod
    def load(path):
        """
        Load agent from a file.
        """
        pass

# Advantage Actor Critic (A2C) with Monte Carlo sampling
class A2C(Agent):
    def __init__(self, env, actor_critic, optimizer, gamma, critic_coef, entropy_coef):
        super().__init__(env)

        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.critic_coef = critic_coef
        self.critic_loss_fn = nn.MSELoss()
        self.entropy_coef = entropy_coef
    
    def select_action(self, observation):
        obs_tensor = torch.from_numpy(np.array(observation)).unsqueeze(0).float()
        probs, _ = self.actor_critic(obs_tensor)
        return torch.argmax(probs, dim=1).item()
    
    def sample_action(self, observation):
        obs_tensor = torch.from_numpy(np.array(observation)).unsqueeze(0).float()
        probs, value = self.actor_critic(obs_tensor)

        action = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log(probs[0, action] + 1e-10)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        return log_prob, action, value.squeeze(), entropy
    
    def train(self, episodes, stats_interval):
        losses = []
        rewards = []
        max_reward = 0

        for e in range(episodes):
            observation, _ = self.env.reset()

            log_probs = []
            values = []
            entropies = []
            episode_rewards = []
            total_episode_reward = 0

            episode_over = False
            while not episode_over:

                log_prob, action, value, entropy = self.sample_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)

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
                R = r + self.gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns).float()
            if returns.size(dim=-1) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            values = torch.stack(values)
            log_probs = torch.stack(log_probs)
            entropies = torch.stack(entropies)
            
            advantages = returns - values
            if advantages.size(dim=-1) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = -(log_probs * advantages.detach()).sum()
            critic_loss = self.critic_loss_fn(values, returns)
            entropy_bonus = entropies.mean()

            loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_bonus
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()

            losses.append(loss.item())

            if (e + 1) % stats_interval == 0:
                print(f"Episodes {(e - stats_interval + 1, e)}:", end=" ")
                print(f"Avg Loss- {sum(losses[e-stats_interval+1:e+1])/stats_interval}", end=" ")
                print(f"Avg Reward- {sum(rewards[e-stats_interval+1:e+1])/stats_interval}")
                
        self.env.close()

        self.episodes_trained += episodes

        return rewards, losses
    
    def save(self, path):
        checkpoint = {
            'env': self.env,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'episodes_trained': self.episodes_trained
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(path):
        checkpoint = torch.load(path)
        a2c = A2C(
            env=checkpoint['env'],
            actor_critic=checkpoint['actor_critic_state_dict'].__class__(),
            optimizer=torch.optim.Adam(checkpoint['actor_critic_state_dict'].__class__().parameters()),
            gamma=checkpoint['gamma']
        )
        a2c.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        a2c.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        a2c.episodes_trained = checkpoint['episodes_trained']

        print(f"Model loaded from {path}")
        return a2c

    
# DDQN (Double DQN) with Temporal Difference (TD)
class DDQN(Agent):
    def __init__(self, env, q_net_class, buffer_size, batch_size,
                 gamma, lr, epsilon_start, epsilon_end, 
                 epsilon_decay, target_update_interval):

        super().__init__(env)

        self.q_net = q_net_class()
        self.target_net = q_net_class()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.target_update_interval = target_update_interval
        self.learn_step = 0

        self.losses = []
    
    def select_action(self, observation):
        q_vals = self.q_net(torch.from_numpy(np.array(observation)).unsqueeze(0).float())
        return torch.argmax(q_vals, dim=1).item()

    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.from_numpy(np.array(state)).unsqueeze(0).float()
        with torch.no_grad():
            q_values = self.q_net(state)
        return torch.argmax(q_values).item()

    def sample_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        # convert (s, a, r, s, d) list into tuples of all s, a, r, s, d
        state, action, reward, next_state, done = zip(*batch)

        return (torch.from_numpy(np.array(state)).float(),      # batch x state size
                torch.tensor(action).view(-1, 1),               # batch x 1
                torch.tensor(reward).view(-1, 1).float(),       # batch x 1
                torch.from_numpy(np.array(next_state)).float(), # batch x state size
                torch.tensor(done).view(-1, 1).float())         # batch x 1

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
        loss = self.loss_fn(q_values, target_q.detach())
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_step % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.learn_step += 1

    def train(self, episodes, stats_interval):
        rewards = []
        max_reward = 0

        for e in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            episode_over = False
            while not episode_over:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_over = terminated or truncated

                self.buffer.append((state, action, reward, next_state, episode_over))
                self.train_step()

                state = next_state
                episode_reward += reward

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            rewards.append(episode_reward)

            if episode_reward > max_reward:
                max_reward = episode_reward
                print(f"New max of {max_reward} in episode {e + 1}")

            if (e + 1) % stats_interval == 0:
                print(f"Episodes {(e - stats_interval + 1, e)}:", end=" ")
                print(f"Avg Reward- {np.mean(rewards[-stats_interval:])}", end=" ")
                print(f"Epsilon- {self.epsilon}")
            
            self.episodes_trained += 1

        return rewards, self.losses

    def save(self, path):
        checkpoint = {
            'env': self.env,
            'q_net_class': self.q_net.__class__,
            'buffer_size': self.buffer.maxlen,
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer': list(self.buffer),  # convert deque to list for saving
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'target_update_interval': self.target_update_interval,
            'learn_step': self.learn_step,
            'episodes_trained': self.episodes_trained
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(path):
        checkpoint = torch.load(path)
        ddqn = DDQN(
            env=checkpoint['env'],
            q_net_class=checkpoint['q_net_class'],
            buffer_size=checkpoint['buffer_size'],
            batch_size=checkpoint['batch_size'],
            gamma=checkpoint['gamma'],
            lr=checkpoint['optimizer_state_dict']['param_groups'][0]['lr'],
            epsilon_start=checkpoint['epsilon'],
            epsilon_end=checkpoint['epsilon_end'],
            epsilon_decay=checkpoint['epsilon_decay'],
            target_update_interval=checkpoint['target_update_interval']
        )
        ddqn.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        ddqn.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        ddqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ddqn.buffer = deque(checkpoint['buffer'], maxlen=checkpoint['buffer_size'])
        ddqn.learn_step = checkpoint['learn_step']
        ddqn.episodes_trained = checkpoint['episodes_trained']

        print(f"Model loaded from {path}")
        return ddqn
    
# Q-Learning with Temporal Difference (TD)
class QLearn(Agent):
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay, min_epsilon):
        super().__init__(env)
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # exploration rate
        self.epsilon_decay = epsilon_decay # exploration decay rate
        self.min_epsilon = min_epsilon # minimum exploration rate

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def select_action(self, observation):
        return np.argmax(self.q_table[observation])

    def epsilon_greedy_policy(self, observation):
        if random.uniform(0, 1) <= self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[observation])

    def train(self, episodes, stats_interval):
        rewards = []
        max_reward = 0

        for e in range(episodes):
            observation, _ = self.env.reset()

            episode_reward = 0

            episode_over = False
            while not episode_over:

                action = self.epsilon_greedy_policy(observation)

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
            
            self.episodes_trained += 1
                
        self.env.close()

        return rewards

    def save(self, path):
        checkpoint = {
            'env': self.env,
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'episodes_trained': self.episodes_trained
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(path):
        checkpoint = torch.load(path)
        qlearn = QLearn(
            env=checkpoint['env'],
            alpha=checkpoint['alpha'],
            gamma=checkpoint['gamma'],
            epsilon=checkpoint['epsilon'],
            epsilon_decay=checkpoint['epsilon_decay'],
            min_epsilon=checkpoint['min_epsilon']
        )
        qlearn.q_table = checkpoint['q_table']
        qlearn.episodes_trained = checkpoint['episodes_trained']

        print(f"Model loaded from {path}")
        return qlearn

# REINFORCE (policy-gradient) for discrete action space with Monte Carlo Sampling
class Reinforce(Agent):
    def __init__(self, env, policy, optimizer, discount):
        super().__init__(env)

        self.policy = policy
        self.optimizer = optimizer
        self.discount = discount

    def sample_action(self, observation):
        probs = self.policy(torch.from_numpy(np.array(observation)).unsqueeze(0).float())
        return probs, torch.multinomial(probs, num_samples=1).item()
    
    def select_action(self, observation):
        probs = self.policy(torch.from_numpy(np.array(observation)).unsqueeze(0).float())
        return torch.argmax(probs, dim=1).item()
    
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

                probs, action = self.sample_action(observation)
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
            for log_prob, R in zip(log_probs, returns):
                loss += -log_prob * R
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

            if (e + 1) % stats_interval == 0:
                print(f"Episodes {(e - stats_interval + 1, e)}:", end=" ")
                print(f"Avg Loss- {sum(losses[e-stats_interval+1:e+1])/stats_interval}", end=" ")
                print(f"Avg Reward- {sum(rewards[e-stats_interval+1:e+1])/stats_interval}")
                
        self.env.close()

        self.episodes_trained += episodes

        return rewards, losses
    
    def save(self, path):
        checkpoint = {
            'env': self.env,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'discount': self.discount,
            'episodes_trained': self.episodes_trained
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(path):
        checkpoint = torch.load(path)
        reinforce = Reinforce(
            env=checkpoint['env'],
            policy=checkpoint['policy_state_dict'].__class__(),
            optimizer=torch.optim.Adam(checkpoint['policy_state_dict'].__class__().parameters()),
            discount=checkpoint['discount']
        )
        reinforce.policy.load_state_dict(checkpoint['policy_state_dict'])
        reinforce.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        reinforce.episodes_trained = checkpoint['episodes_trained']

        print(f"Model loaded from {path}")
        return reinforce