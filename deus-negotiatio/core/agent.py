import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
import numpy as np
import random
import os

from core.dqn_network import DuelingDQN
from core.state_encoder import StateEncoder
from core.reward_function import RewardFunction
from learning.experience_buffer import PrioritizedReplayBuffer

class DeusNegotiatioAgent:
    """
    The central agent for a single intersection.
    Integrates:
    - Sensing (StateEncoder)
    - Decision Making (DQN)
    - Learning (Optimizer + ReplayBuffer)
    """
    def __init__(self, agent_id, config, state_dim, action_dim, device='cpu'):
        self.agent_id = agent_id
        self.config = config
        self.device = device
        
        # Modules
        self.encoder = StateEncoder(config)
        self.reward_fn = RewardFunction()
        
        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.get('learning_rate', 1e-4))
        
        # Memory
        self.memory = PrioritizedReplayBuffer(config.get('buffer_size', 100000))
        
        # Hyperparameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.steps = 0
        
    def select_action(self, state, valid_actions=None):
        """
        Epsilon-greedy action selection.
        valid_actions: list of valid action indices (optional masking)
        """
        self.steps += 1
        
        if random.random() < self.epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            # Assuming action dim is output dim of network
            return random.randint(0, self.policy_net.output_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            
            if valid_actions:
                # Mask invalid actions with -inf
                full_mask = torch.full_like(q_values, float('-inf'))
                full_mask[0, valid_actions] = 0
                q_values = q_values + full_mask
                
            return q_values.argmax(dim=1).item()

    def update_policy(self):
        """
        Performs one step of optimization using a batch from replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        # Unpack batch
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(np.array(batch[1])).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch[2])).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch[4])).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a)
        # Gather Q values for specific actions taken
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for next states (Double DQN usually)
        # Here implementing DDQN: action selection from policy net, evaluation from target net
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_net(next_state_batch).gather(1, next_actions)
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Loss
        loss = (current_q - target_q).pow(2) * weights_t
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities
        td_errors = (current_q - target_q).detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-5
        self.memory.update_priorities(indices, new_priorities.flatten())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
