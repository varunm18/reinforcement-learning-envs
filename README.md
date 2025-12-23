# Reinforcement Learning
Various RL algos written from scratch and used on gymnasium environments

## Details

### Algorithms Created (in trainer.py)
* Q-Learning
* Double-Deep Q-Network (DDQN)
* REINFORCE
* Advantage Actor Critic (A2C)
* Proximal Policy Optimization (PPO)

All algorithms use Monte Carlo sampling except for Q-Learning and DDQN which use Temporal Difference (TD(0))


## Results

| Taxi (Q-Learning) | Cart Pole (REINFORCE) | LunarLander (PPO) |
|------|------|------|
|<img src="https://github.com/user-attachments/assets/876bce7e-995c-4393-9f58-27c08faedc0a" width="100%" height="100%"/>|<img src="https://github.com/user-attachments/assets/68911fe0-fde4-4ab8-abf0-740070250bd3" width="100%" height="100%"/>|<img src="https://github.com/user-attachments/assets/9f2b7729-ccfd-4dec-9192-79dd1cdece85" width="100%" height="100%"/>|

| Space Invaders (DDQN, to be improved) | Pong (To Do) | Car Racer (To Do) |
|------|------|------|
|<img src="https://github.com/user-attachments/assets/70e590e7-c949-491b-a633-6775c844c0b3" width="100%" height="100%"/>|||

## TODO

- [x] Change REINFORCE class to be compatible with CNNs (view to squeeze)
- [x] Merge all classes under super class
- [ ] Solve Car-racing
- [ ] Solve Space-invaders
- [ ] Solve Pong
- [ ] Get better results for Blackjack
