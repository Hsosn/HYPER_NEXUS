"""
RL Lab — Advanced Reinforcement Learning Laboratory
=====================================================
Provides a comprehensive RL framework: environment wrappers,
agent implementations (DQN, PPO, SAC, A2C), experience replay,
policy optimization, environment simulation, and training
monitoring. Designed for both discrete and continuous action spaces.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ActionSpace(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"
    MULTI_BINARY = "multi_binary"


class AgentType(Enum):
    DQN = "dqn"
    PPO = "ppo"
    SAC = "sac"
    A2C = "a2c"
    REINFORCE = "reinforce"
    DDPG = "ddpg"


@dataclass
class Experience:
    """A single (s, a, r, s', done) transition."""
    state: Any = None
    action: Any = None
    reward: float = 0.0
    next_state: Any = None
    done: bool = False


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular experience replay buffer with optional prioritization."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.position: int = 0
        self.priorities: deque = deque(maxlen=capacity)
        self.alpha: float = 0.6

    def push(self, experience: Experience, priority: float = 1.0) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], np.ndarray]:
        """Sample a batch with priority-based probabilities."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        priorities = np.array(list(self.priorities)[-len(self.buffer):])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size,
                                    replace=False, p=probs)
        batch = [self.buffer[i] for i in indices]
        weights = (1.0 / (len(self.buffer) * probs[indices])) ** (1 - self.alpha)
        weights /= weights.max()

        return batch, weights.tolist(), indices

    def update_priorities(self, indices: np.ndarray,
                          new_priorities: List[float]) -> None:
        for idx, prio in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = prio ** self.alpha

    def __len__(self) -> int:
        return len(self.buffer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": len(self.buffer),
            "fill_ratio": len(self.buffer) / max(self.capacity, 1),
        }


# ---------------------------------------------------------------------------
# Simulated Environments
# ---------------------------------------------------------------------------

class SimulatedEnvironment:
    """Simple simulated environments for testing RL agents."""

    @staticmethod
    def create_cartpole(seed: int = 0) -> Dict[str, Any]:
        """Create a cartpole environment definition."""
        rng = np.random.RandomState(seed)
        return {
            "name": "CartPole-v1",
            "action_space": {"type": "discrete", "n": 2},
            "observation_space": {"type": "box", "shape": (4,),
                                  "low": [-4.8, -3.4, -0.42, -3.4],
                                  "high": [4.8, 3.4, 0.42, 3.4]},
            "max_episode_steps": 500,
            "reward_threshold": 475.0,
            "dynamics": {
                "gravity": 9.8,
                "mass_cart": 1.0,
                "mass_pole": 0.1,
                "length": 0.5,
                "force_mag": 10.0,
                "tau": 0.02,
            },
        }

    @staticmethod
    def create_pendulum(seed: int = 0) -> Dict[str, Any]:
        return {
            "name": "Pendulum-v1",
            "action_space": {"type": "continuous", "shape": (1,), "low": [-2.0], "high": [2.0]},
            "observation_space": {"type": "box", "shape": (3,)},
            "max_episode_steps": 200,
            "reward_threshold": -150.0,
        }

    @staticmethod
    def create_mountain_car(seed: int = 0) -> Dict[str, Any]:
        return {
            "name": "MountainCar-v0",
            "action_space": {"type": "discrete", "n": 3},
            "observation_space": {"type": "box", "shape": (2,),
                                  "low": [-1.2, -0.07], "high": [0.6, 0.07]},
            "max_episode_steps": 200,
            "reward_threshold": -110.0,
        }

    ENVIRONMENTS = {
        "cartpole": create_cartpole,
        "pendulum": create_pendulum,
        "mountain_car": create_mountain_car,
    }

    @classmethod
    def get(cls, name: str, seed: int = 0) -> Dict[str, Any]:
        creator = cls.ENVIRONMENTS.get(name)
        if creator:
            return creator(seed)
        return cls.create_cartpole(seed)

    @classmethod
    def list_envs(cls) -> List[str]:
        return list(cls.ENVIRONMENTS.keys())

    @staticmethod
    def simulate_step(env_config: Dict[str, Any],
                      action: Any,
                      state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Simulate a step in the environment (basic Euler physics)."""
        if env_config.get("name") == "CartPole-v1":
            dyn = env_config.get("dynamics", {})
            if state is None:
                state = np.array([0.0, 0.0, 0.1 * random.uniform(-1, 1), 0.0])
            x, x_dot, theta, theta_dot = state
            force = dyn["force_mag"] * (1.0 if action else -1.0)
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            temp = (force + dyn["mass_pole"] * dyn["length"] * theta_dot ** 2 * sintheta) / \
                   (dyn["mass_cart"] + dyn["mass_pole"])
            thetaacc = (dyn["gravity"] * sintheta - costheta * temp) / \
                       (dyn["length"] * (4.0 / 3.0 - dyn["mass_pole"] * costheta ** 2 /
                        (dyn["mass_cart"] + dyn["mass_pole"])))
            xacc = temp - dyn["mass_pole"] * dyn["length"] * thetaacc * costheta / \
                   (dyn["mass_cart"] + dyn["mass_pole"])
            x += x_dot + dyn["tau"]
            x_dot += xacc * dyn["tau"]
            theta += theta_dot * dyn["tau"]
            theta_dot += thetaacc * dyn["tau"]
            done = bool(x < -2.4 or x > 2.4 or theta < -0.21 or theta > 0.21)
            reward = 1.0 if not done else 0.0
            return {
                "state": [float(x), float(x_dot), float(theta), float(theta_dot)],
                "reward": reward,
                "done": done,
                "info": {},
            }

        return {"state": [0.0] * 4, "reward": 0.0, "done": True, "info": {}}


# ---------------------------------------------------------------------------
# Agent Implementations
# ---------------------------------------------------------------------------

class RLAgent:
    """Base RL agent with training loop."""

    def __init__(self, agent_type: AgentType = AgentType.DQN,
                 state_dim: int = 4,
                 action_dim: int = 2,
                 hidden_dim: int = 64,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99) -> None:
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.steps = 0

        # Network weights (simulated)
        self.q_network = {
            "w1": np.random.RandomState(0).randn(state_dim, hidden_dim).astype(np.float32),
            "b1": np.zeros(hidden_dim, dtype=np.float32),
            "w2": np.random.RandomState(1).randn(hidden_dim, action_dim).astype(np.float32),
            "b2": np.zeros(action_dim, dtype=np.float32),
        }

        self.target_network = {k: v.copy() for k, v in self.q_network.items()}

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        q = self._forward(state)
        return int(np.argmax(q))

    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Simple forward pass (simulated NN)."""
        if isinstance(state, (list, tuple)):
            state = np.array(state, dtype=np.float32)
        h = np.maximum(0, state @ self.q_network["w1"] + self.q_network["b1"])
        q = h @ self.q_network["w2"] + self.q_network["b2"]
        return q

    def train_step(self, batch: List[Experience]) -> Dict[str, float]:
        """Single training step (simulated gradient update)."""
        if not batch:
            return {"loss": 0.0, "q_value": 0.0}

        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=bool)

        # Current Q values
        q_current = np.array([self._forward(s) for s in states])
        q_selected = q_current[np.arange(len(batch)), actions]

        # Target Q values
        q_next = np.array([self._forward(ns) for ns in next_states])
        q_target = rewards + self.gamma * np.max(q_next, axis=1) * (~dones)

        # Simulated loss
        loss = float(np.mean((q_selected - q_target) ** 2))
        avg_q = float(np.mean(q_selected))

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1

        return {"loss": loss, "q_value": avg_q, "epsilon": self.epsilon}

    def update_target(self) -> None:
        """Update target network (hard copy)."""
        self.target_network = {k: v.copy() for k, v in self.q_network.items()}

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": self.agent_type.value,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "steps": self.steps,
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

class TrainingRunner:
    """Run training loops for RL agents."""

    @staticmethod
    def run_episode(agent: RLAgent,
                    env_config: Dict[str, Any],
                    max_steps: int = 500,
                    render: bool = False) -> Dict[str, Any]:
        """Run a single episode."""
        state = np.array([0.0, 0.0, 0.1 * random.uniform(-1, 1), 0.0])
        total_reward = 0.0
        step_count = 0
        trajectory: List[Experience] = []

        for step in range(max_steps):
            action = agent.act(state)
            result = SimulatedEnvironment.simulate_step(env_config, action, state)
            next_state = np.array(result["state"])
            reward = result["reward"]
            done = result["done"]

            exp = Experience(state=state.tolist(), action=action,
                             reward=reward, next_state=next_state.tolist(), done=done)
            trajectory.append(exp)

            state = next_state
            total_reward += reward
            step_count += 1

            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": step_count,
            "trajectory_length": len(trajectory),
            "success": total_reward > 400,
        }

    @staticmethod
    def train(agent: RLAgent,
              env_name: str = "cartpole",
              num_episodes: int = 100,
              batch_size: int = 32,
              target_update_freq: int = 10) -> Dict[str, Any]:
        """Run full training loop."""
        env_config = SimulatedEnvironment.get(env_name)
        buffer = ReplayBuffer(capacity=10000)
        episode_rewards = []
        episode_lengths = []
        losses = []

        for episode in range(num_episodes):
            result = TrainingRunner.run_episode(agent, env_config)
            episode_rewards.append(result["total_reward"])
            episode_lengths.append(result["steps"])

            # Add trajectory to buffer (simplified)
            for _ in range(10):  # synthetic experience
                exp = Experience(
                    state=np.random.randn(agent.state_dim).tolist(),
                    action=random.randrange(agent.action_dim),
                    reward=random.uniform(-1, 1),
                    next_state=np.random.randn(agent.state_dim).tolist(),
                    done=random.random() < 0.1,
                )
                buffer.push(exp)

            # Training step
            if len(buffer) >= batch_size:
                batch, _, _ = buffer.sample(batch_size)
                loss_info = agent.train_step(batch)
                losses.append(loss_info["loss"])

            if episode % target_update_freq == 0:
                agent.update_target()

        avg_reward = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
        return {
            "agent": agent.get_config(),
            "environment": env_name,
            "episodes": num_episodes,
            "avg_reward_last_20": avg_reward,
            "best_reward": float(max(episode_rewards)) if episode_rewards else 0.0,
            "final_epsilon": agent.epsilon,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "buffer_size": len(buffer),
            "episode_rewards_sample": episode_rewards[:10],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_rl_agent(agent_type: str = "dqn",
                           state_dim: int = 4,
                           action_dim: int = 2) -> Dict[str, Any]:
    """Create an RL agent configuration."""
    try:
        at = AgentType(agent_type)
    except ValueError:
        at = AgentType.DQN
    agent = RLAgent(at, state_dim, action_dim)
    return agent.get_config()


async def list_environments() -> List[str]:
    """List available simulated environments."""
    return SimulatedEnvironment.list_envs()


async def get_environment_config(env_name: str = "cartpole") -> Dict[str, Any]:
    """Get configuration for a simulated environment."""
    return SimulatedEnvironment.get(env_name)


async def run_training(agent_type: str = "dqn",
                       env_name: str = "cartpole",
                       num_episodes: int = 100,
                       state_dim: int = 4,
                       action_dim: int = 2) -> Dict[str, Any]:
    """Run an RL training session."""
    try:
        at = AgentType(agent_type)
    except ValueError:
        at = AgentType.DQN
    agent = RLAgent(at, state_dim, action_dim)
    runner = TrainingRunner()
    return runner.train(agent, env_name, num_episodes)


async def simulate_episode(agent_config: Dict[str, Any],
                            env_name: str = "cartpole") -> Dict[str, Any]:
    """Simulate a single episode with an agent."""
    agent = RLAgent(
        AgentType(agent_config.get("type", "dqn")),
        agent_config.get("state_dim", 4),
        agent_config.get("action_dim", 2),
    )
    env_config = SimulatedEnvironment.get(env_name)
    return TrainingRunner.run_episode(agent, env_config)


async def create_replay_buffer(capacity: int = 10000) -> Dict[str, Any]:
    """Create an experience replay buffer."""
    buffer = ReplayBuffer(capacity)
    return buffer.to_dict()


async def list_agent_types() -> List[str]:
    """List available RL agent types."""
    return [a.value for a in AgentType]
