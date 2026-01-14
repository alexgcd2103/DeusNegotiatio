import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.agent import DeusNegotiatioAgent

class TrainingLoop:
    """
    Manages the recursive learning cycle.
    """
    def __init__(self, agent: DeusNegotiatioAgent, env, config):
        self.agent = agent
        self.env = env # Simulation environment wrapper (e.g. SUMO)
        self.config = config
        self.episodes = config.get('num_episodes', 100)
        self.steps_per_episode = config.get('steps_per_episode', 1000)
        
    def run(self):
        print(f"Starting training on device: {self.agent.device}")
        print(f"Episodes: {self.episodes}, Steps per episode: {self.steps_per_episode}")
        
        best_reward = float('-inf')
        reward_history = []
        
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(self.steps_per_episode):
                # 1. Select Action
                action = self.agent.select_action(state)
                
                # 2. Step Environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # 3. Store Experience
                self.agent.memory.push(state, action, reward, next_state, done)
                
                # 4. Train
                loss = self.agent.update_policy()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Track rewards
            reward_history.append(episode_reward)
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save("best_model.pth")
            
            # 5. Sync Target Network periodically
            if episode % self.config.get('target_update_interval', 10) == 0:
                self.agent.sync_target_network()
            
            # Calculate metrics
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            avg_reward_10 = sum(reward_history[-10:]) / min(len(reward_history), 10)
            
            print(f"Episode {episode:3d} | Reward: {episode_reward:8.2f} | "
                  f"Avg(10): {avg_reward_10:8.2f} | Best: {best_reward:8.2f} | "
                  f"Epsilon: {self.agent.epsilon:.4f} | Loss: {avg_loss:.4f}")

        # Save final model
        self.agent.save("final_model.pth")
        print(f"\nTraining complete! Best reward: {best_reward:.2f}")

