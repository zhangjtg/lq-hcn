"""
DOREA: Deep Offline Reinforcement Learning Negotiating Agent Framework
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


# ============================================================================
# Neural Network Components
# ============================================================================

class PolicyNetwork(nn.Module):
	"""Stochastic policy network for SAC-based agent"""

	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
		super().__init__()

		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)

		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)

		self.action_dim = action_dim

	def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward pass returning mean and log_std"""
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))

		mean = self.mean(x)
		log_std = self.log_std(x)
		log_std = torch.clamp(log_std, -20, 2)

		return mean, log_std

	def sample(self, state: torch.Tensor, epsilon: float = 1e-6):
		"""Sample action from the policy"""
		mean, log_std = self.forward(state)
		std = log_std.exp()

		normal = Normal(mean, std)
		z = normal.rsample()
		action = torch.tanh(z)

		# Calculate log probability
		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(dim=-1, keepdim=True)

		return action, log_prob, mean, std


class QNetwork(nn.Module):
	"""Q-value network (critic)"""

	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
		super().__init__()

		self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Forward pass returning Q-value"""
		x = torch.cat([state, action], dim=-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		q_value = self.fc3(x)
		return q_value


class DensityRatioEstimator(nn.Module):
	"""Density ratio estimator for online-ness measurement"""

	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
		super().__init__()

		self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		"""Forward pass returning density ratio"""
		x = torch.cat([state, action], dim=-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		ratio = self.fc3(x)
		return ratio


# ============================================================================
# Replay Buffer Components
# ============================================================================

class ReplayBuffer:
	"""Standard replay buffer for storing transitions"""

	def __init__(self, capacity: int = int(2e6)):
		self.buffer = deque(maxlen=capacity)

	def add(self, state, action, reward, next_state, done):
		"""Add transition to buffer"""
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size: int):
		"""Sample batch from buffer"""
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		return (
			np.array(states),
			np.array(actions),
			np.array(rewards).reshape(-1, 1),
			np.array(next_states),
			np.array(dones).reshape(-1, 1)
		)

	def size(self):
		return len(self.buffer)


class PrioritizedBuffer:
	"""Prioritized replay buffer using density ratio for fine-tuning"""

	def __init__(self, capacity: int = int(2e6)):
		self.offline_buffer = deque(maxlen=capacity)
		self.online_buffer = deque(maxlen=capacity)
		self.priorities = []

	def add_offline(self, state, action, reward, next_state, done):
		"""Add offline transition, default density_ratio to 0 (low online-ness)"""
		# 存储格式: (state, action, reward, next_state, done, density_ratio)
		self.offline_buffer.append((state, action, reward, next_state, done, 0.0))

	def add_online(self, state, action, reward, next_state, done, density_ratio):
		"""Add online transition with density ratio"""
		# 存储格式: (state, action, reward, next_state, done, density_ratio)
		self.online_buffer.append((state, action, reward, next_state, done, density_ratio))

	def sample_balanced(self, batch_size: int):
		"""Sample batch using balanced experience replay with weighted online sampling"""
		# 采样比例：一半在线，一半离线
		online_size = min(batch_size // 2, len(self.online_buffer))
		offline_size = batch_size - online_size
		batch = []

		# 1. 从 Online Buffer 进行加权采样
		if online_size > 0:
			online_list = list(self.online_buffer)
			if online_list:
				# 提取存储的密度比率 (元组的第6个元素，索引为5)
				weights = np.array([t[5] for t in online_list])

				# 确保权重为正数并归一化，形成概率分布
				weights = np.clip(weights, 1e-6, None)
				weights_sum = np.sum(weights)
				if weights_sum > 0:
					weights = weights / weights_sum
				else:
					weights = np.ones(len(online_list)) / len(online_list)

				# 根据权重采样索引
				indices = np.random.choice(len(online_list), size=online_size, p=weights, replace=True)

				for idx in indices:
					batch.append(online_list[idx])

		# 2. 从 Offline Buffer 均匀采样
		if offline_size > 0:
			offline_samples = random.sample(list(self.offline_buffer), offline_size)
			batch.extend(offline_samples)

		# 3. 解包并返回标准格式 (s, a, r, s', d)，去除存储的 density_ratio
		if not batch:
			return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

		transposed = list(zip(*batch))
		return (
			np.array(transposed[0]),  # states
			np.array(transposed[1]),  # actions
			np.array(transposed[2]).reshape(-1, 1),  # rewards
			np.array(transposed[3]),  # next_states
			np.array(transposed[4]).reshape(-1, 1)  # dones
		)

	def size(self):
		return len(self.offline_buffer) + len(self.online_buffer)



# ============================================================================
# DOREA Agent Components
# ============================================================================

class CQLAgent:
	"""Conservative Q-Learning agent for offline RL"""

	def __init__(
			self,
			state_dim: int,
			action_dim: int,
			hidden_dim: int = 256,
			lr_actor: float = 1e-4,
			lr_critic: float = 3e-4,
			gamma: float = 0.99,
			tau: float = 0.005,
			alpha: float = 0.2,
			cql_alpha: float = 1.0,
			device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
	):
		self.device = device
		self.gamma = gamma
		self.tau = tau
		self.alpha = alpha
		self.cql_alpha = cql_alpha

		# Initialize networks
		self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

		self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
		self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

		self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
		self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)

		# Copy parameters to target networks
		self.q1_target.load_state_dict(self.q1.state_dict())
		self.q2_target.load_state_dict(self.q2.state_dict())

		# Initialize optimizers
		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
		self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr_critic)
		self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr_critic)

		# Auto-tune temperature
		self.target_entropy = -action_dim
		self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)

	def select_action(self, state: np.ndarray, deterministic: bool = False):
		"""Select action from policy"""
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

		with torch.no_grad():
			if deterministic:
				mean, _ = self.policy(state)
				action = torch.tanh(mean)
			else:
				action, _, _, _ = self.policy.sample(state)

		return action.cpu().numpy()[0]

	def update_critics(
			self,
			states: torch.Tensor,
			actions: torch.Tensor,
			rewards: torch.Tensor,
			next_states: torch.Tensor,
			dones: torch.Tensor
	):
		"""Update Q-networks with CQL regularizer"""

		# Sample actions from current policy
		with torch.no_grad():
			next_actions, next_log_probs, _, _ = self.policy.sample(next_states)

			# Target Q-values
			target_q1 = self.q1_target(next_states, next_actions)
			target_q2 = self.q2_target(next_states, next_actions)
			target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs

			target_q = rewards + (1 - dones) * self.gamma * target_q

		# Current Q-values
		current_q1 = self.q1(states, actions)
		current_q2 = self.q2(states, actions)

		# CQL regularizer
		random_actions = torch.FloatTensor(
			states.shape[0], actions.shape[1]
		).uniform_(-1, 1).to(self.device)

		random_q1 = self.q1(states, random_actions)
		random_q2 = self.q2(states, random_actions)

		# Sample actions from current policy
		policy_actions, _, _, _ = self.policy.sample(states)
		policy_actions = policy_actions.detach()
		policy_q1 = self.q1(states, policy_actions)
		policy_q2 = self.q2(states, policy_actions)

		# CQL loss (conservative regularizer)
		cql1_loss = (torch.logsumexp(
			torch.cat([random_q1, policy_q1], dim=1), dim=1
		).mean() - current_q1.mean())

		cql2_loss = (torch.logsumexp(
			torch.cat([random_q2, policy_q2], dim=1), dim=1
		).mean() - current_q2.mean())

		# Total Q loss
		q1_loss = F.mse_loss(current_q1, target_q) + self.cql_alpha * cql1_loss
		q2_loss = F.mse_loss(current_q2, target_q) + self.cql_alpha * cql2_loss

		# Update Q-networks
		self.q1_optimizer.zero_grad()
		q1_loss.backward()
		self.q1_optimizer.step()

		self.q2_optimizer.zero_grad()
		q2_loss.backward()
		self.q2_optimizer.step()

		return q1_loss.item(), q2_loss.item()

	def update_policy(self, states: torch.Tensor):
		"""Update policy network"""

		actions, log_probs, _, _ = self.policy.sample(states)

		q1_pi = self.q1(states, actions)
		q2_pi = self.q2(states, actions)
		min_q_pi = torch.min(q1_pi, q2_pi)

		policy_loss = (self.alpha * log_probs - min_q_pi).mean()

		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()

		# Update temperature
		alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

		self.alpha_optimizer.zero_grad()
		alpha_loss.backward()
		self.alpha_optimizer.step()

		self.alpha = self.log_alpha.exp().item()

		return policy_loss.item(), alpha_loss.item()

	def soft_update_target(self):
		"""Soft update of target networks"""
		for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class EnsembleCQLAgent:
	"""Ensemble of CQL agents for DOREA"""

	def __init__(
			self,
			state_dim: int,
			action_dim: int,
			n_ensemble: int = 5,
			**kwargs
	):
		self.n_ensemble = n_ensemble
		self.agents = [
			CQLAgent(state_dim, action_dim, **kwargs)
			for _ in range(n_ensemble)
		]

	def select_action(self, state: np.ndarray, deterministic: bool = False):
		"""Select action using ensemble average"""
		actions = []
		for agent in self.agents:
			action = agent.select_action(state, deterministic)
			actions.append(action)

		# Return mean action
		return np.mean(actions, axis=0)

	def update(self, batch):
		"""Update all agents in ensemble"""
		states, actions, rewards, next_states, dones = batch

		states = torch.FloatTensor(states).to(self.agents[0].device)
		actions = torch.FloatTensor(actions).to(self.agents[0].device)
		rewards = torch.FloatTensor(rewards).to(self.agents[0].device)
		next_states = torch.FloatTensor(next_states).to(self.agents[0].device)
		dones = torch.FloatTensor(dones).to(self.agents[0].device)

		losses = []
		for agent in self.agents:
			q1_loss, q2_loss = agent.update_critics(states, actions, rewards, next_states, dones)
			policy_loss, alpha_loss = agent.update_policy(states)
			agent.soft_update_target()

			losses.append({
				'q1_loss': q1_loss,
				'q2_loss': q2_loss,
				'policy_loss': policy_loss,
				'alpha_loss': alpha_loss
			})

		return losses


# ============================================================================
# DOREA Framework
# ============================================================================

class DOREAFramework:
	"""
	Deep Offline Reinforcement Learning Negotiating Agent Framework

	Main framework implementing offline learning and strategy fine-tuning.
	"""

	def __init__(
			self,
			state_dim: int,
			action_dim: int,
			n_ensemble: int = 5,
			batch_size: int = 256,
			**kwargs
	):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.batch_size = batch_size

		# Initialize ensemble agent
		self.agent = EnsembleCQLAgent(state_dim, action_dim, n_ensemble, **kwargs)

		# Initialize buffers
		self.offline_buffer = ReplayBuffer()
		self.prioritized_buffer = PrioritizedBuffer()

		# Density ratio estimator for fine-tuning
		self.density_estimator = DensityRatioEstimator(
			state_dim, action_dim
		).to(self.agent.agents[0].device)

		self.density_optimizer = torch.optim.Adam(
			self.density_estimator.parameters(), lr=3e-4
		)

		self.is_finetuning = False

	def add_offline_data(self, state, action, reward, next_state, done):
		"""Add offline transition data"""
		self.offline_buffer.add(state, action, reward, next_state, done)
		self.prioritized_buffer.add_offline(state, action, reward, next_state, done)

	def add_online_data(self, state, action, reward, next_state, done, density_ratio=None):
		"""
		Add online transition data for fine-tuning (Section 4.2).

		In DOREA, online transitions are stored with a density ratio w(s,a)=p_on(s,a)/p_off(s,a),
		which is used by balanced experience replay (Eq. 12-13).

		- If `density_ratio` is provided, we store it directly.
		- If `density_ratio` is None, we compute it with the current density estimator.
		"""
		if density_ratio is None:
			try:
				with torch.no_grad():
					st = torch.FloatTensor(state).unsqueeze(0).to(self.agent.agents[0].device)
					ac = torch.FloatTensor(action).unsqueeze(0).to(self.agent.agents[0].device)
					density_ratio = float(self.density_estimator(st, ac).detach().cpu().item())
			except Exception:
				density_ratio = 1.0
		# clip to keep sampling stable
		density_ratio = float(np.clip(float(density_ratio), 1e-6, 1e6))
		self.prioritized_buffer.add_online(state, action, reward, next_state, done, density_ratio)
	'''
	def add_online_data(self, state, action, reward, next_state, done):
		"""Add online transition data for fine-tuning"""
		self.prioritized_buffer.add_online(state, action, reward, next_state, done)
	'''
	def train_offline(self, n_steps: int = int(1e6)):
		"""Train agent using offline data (Section 4.1)"""
		print(f"Training offline for {n_steps} steps...")

		for step in range(n_steps):
			if self.offline_buffer.size() < self.batch_size:
				continue

			batch = self.offline_buffer.sample(self.batch_size)
			losses = self.agent.update(batch)

			if (step + 1) % 10000 == 0:
				avg_q_loss = np.mean([l['q1_loss'] + l['q2_loss'] for l in losses])
				avg_policy_loss = np.mean([l['policy_loss'] for l in losses])
				print(f"Step {step + 1}: Q Loss = {avg_q_loss:.4f}, Policy Loss = {avg_policy_loss:.4f}")

	def update_density_estimator(self, batch_size: int = 256):
		"""Update density ratio estimator (Section 4.2)"""
		if len(self.prioritized_buffer.online_buffer) < batch_size // 2:
			return 0.0

		# Sample from online buffer
		online_samples = random.sample(
			list(self.prioritized_buffer.online_buffer), batch_size // 2
		)
		# Sample from offline buffer
		offline_samples = random.sample(
			list(self.prioritized_buffer.offline_buffer), batch_size // 2
		)

		# Extract states and actions
		online_states = torch.FloatTensor([s[0] for s in online_samples]).to(
			self.agent.agents[0].device
		)
		online_actions = torch.FloatTensor([s[1] for s in online_samples]).to(
			self.agent.agents[0].device
		)

		offline_states = torch.FloatTensor([s[0] for s in offline_samples]).to(
			self.agent.agents[0].device
		)
		offline_actions = torch.FloatTensor([s[1] for s in offline_samples]).to(
			self.agent.agents[0].device
		)

		# Compute density ratios
		online_ratios = self.density_estimator(online_states, online_actions)
		offline_ratios = self.density_estimator(offline_states, offline_actions)

		# Jensen-Shannon divergence loss (Eq. 11)
		def f_prime(w):
			return w / (1 + w)

		def f_star_conjugate(t):
			return -torch.log(2 - torch.exp(t))

		loss = -(f_prime(online_ratios).mean() -
				 f_star_conjugate(f_prime(offline_ratios)).mean())

		self.density_optimizer.zero_grad()
		loss.backward()
		self.density_optimizer.step()

		return loss.item()

	def finetune(self, n_steps: int = 10000):
		"""Fine-tune agent with online data (Section 4.2)"""
		print(f"Fine-tuning for {n_steps} steps...")
		self.is_finetuning = True

		for step in range(n_steps):
			if self.prioritized_buffer.size() < self.batch_size:
				continue

			# Update density estimator
			density_loss = self.update_density_estimator()

			# Sample balanced batch
			batch = self.prioritized_buffer.sample_balanced(self.batch_size)
			losses = self.agent.update(batch)

			if (step + 1) % 1000 == 0:
				avg_q_loss = np.mean([l['q1_loss'] + l['q2_loss'] for l in losses])
				avg_policy_loss = np.mean([l['policy_loss'] for l in losses])
				print(f"Step {step + 1}: Q Loss = {avg_q_loss:.4f}, "
					  f"Policy Loss = {avg_policy_loss:.4f}, "
					  f"Density Loss = {density_loss:.4f}")

	def select_action(self, state: np.ndarray, deterministic: bool = False):
		"""Select action using ensemble policy"""
		return self.agent.select_action(state, deterministic)

	def save(self, path: str):
		"""Save model"""
		torch.save({
			'agents': [agent.policy.state_dict() for agent in self.agent.agents],
			'q_networks': [(agent.q1.state_dict(), agent.q2.state_dict())
						   for agent in self.agent.agents]
		}, path)

	def load(self, path: str):
		"""Load model"""
		checkpoint = torch.load(path)
		for i, agent in enumerate(self.agent.agents):
			agent.policy.load_state_dict(checkpoint['agents'][i])
			agent.q1.load_state_dict(checkpoint['q_networks'][i][0])
			agent.q2.load_state_dict(checkpoint['q_networks'][i][1])


