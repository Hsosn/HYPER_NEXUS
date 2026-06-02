# RL Lab — Advanced Level

**Level:** Advanced  
**Category:** AI/ML

## Overview
Reinforcement Learning experiment toolkit with real PyTorch agent architectures, environment descriptions, and training configurations.

## Capabilities
- **Environments**: GridWorld, CartPole, MountainCar, LunarLander (Gymnasium)
- **Agents**: DQN, REINFORCE, A2C, PPO with real architecture descriptions
- **Training**: Complete training loop patterns with experience replay
- **Evaluation**: Agent evaluation with metrics

## Usage
```bash
python rl_lab.py env --type cartpole --state-dim 4 --action-dim 2
python rl_lab.py agent --type ppo --state-dim 4 --action-dim 2
python rl_lab.py train --algorithm ppo --episodes 1000
python rl_lab.py evaluate --agent agent.pt --episodes 100
```

## Agent Architectures
| Type | Description |
|------|-------------|
| DQN | Deep Q-Network with replay buffer |
| REINFORCE | Policy gradient with baseline |
| A2C | Advantage Actor-Critic with GAE |
| PPO | Proximal Policy Optimization |

## Requirements
- PyTorch, numpy
- (Optional) gymnasium for environments
- (Optional) stable-baselines3 for quick prototyping
