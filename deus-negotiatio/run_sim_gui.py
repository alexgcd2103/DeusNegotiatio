#!/usr/bin/env python3
"""
Oxford-Hyde Park Intersection - GUI Visualization

Run a trained DeusNegotiatio agent with SUMO-GUI visualization.
This allows you to watch the AI control traffic in real-time.

Usage:
    python run_sim_gui.py                    # Run with trained model
    python run_sim_gui.py --baseline         # Run with baseline timing
    python run_sim_gui.py --random           # Run with random actions
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np


def run_gui_simulation(mode='trained', model_path='best_model.pth', episodes=1):
    """
    Run SUMO-GUI simulation with specified control mode.
    
    Args:
        mode: 'trained', 'baseline', or 'random'
        model_path: Path to trained model (for 'trained' mode)
        episodes: Number of episodes to run
    """
    from envs.oxford_hydepark_env import OxfordHydeParkEnv
    from core.agent import DeusNegotiatioAgent
    import yaml
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config/learning_hyperparams.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment with GUI
    env = OxfordHydeParkEnv(use_gui=True)
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"  Oxford-Hyde Park Intersection - GUI Simulation")
    print(f"  Mode: {mode.upper()}")
    print(f"  Device: {device}")
    print(f"  Episodes: {episodes}")
    print(f"{'='*60}\n")
    
    # Load trained agent if using trained mode
    agent = None
    if mode == 'trained':
        agent = DeusNegotiatioAgent(
            agent_id="GUI_Agent",
            config=config,
            state_dim=env.obs_dim,
            action_dim=env.action_space.n,
            device=device
        )
        
        # Load trained weights
        full_model_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_model_path):
            agent.load(full_model_path)
            agent.epsilon = 0.0  # No exploration during visualization
            print(f"  Loaded trained model from: {model_path}")
        else:
            print(f"  WARNING: Model not found at {full_model_path}")
            print(f"  Falling back to random actions")
            mode = 'random'
    
    # Run episodes
    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            # Select action based on mode
            if mode == 'trained' and agent is not None:
                action = agent.select_action(state)
            elif mode == 'baseline':
                # Let SUMO handle timing with baseline program
                action = 0  # Keep current phase
            else:  # random
                action = env.action_space.sample()
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step:4d} | Reward: {episode_reward:8.2f} | "
                      f"Queue: {info['queue_length']:3d} | "
                      f"Vehicles: {info['vehicles_in_sim']:4d}")
            
            if terminated or truncated:
                break
        
        print(f"\n  Episode {episode + 1} Complete!")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Total Wait Time: {info['total_wait_time']:.1f} seconds")
        print(f"  Throughput: {info['throughput']} vehicles")
    
    env.close()
    print(f"\n{'='*60}")
    print("  Simulation Complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Oxford-Hyde Park simulation with GUI")
    parser.add_argument('--baseline', action='store_true', 
                        help='Use baseline timing (no AI)')
    parser.add_argument('--random', action='store_true',
                        help='Use random actions')
    parser.add_argument('--model', type=str, default='best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run')
    
    args = parser.parse_args()
    
    if args.baseline:
        mode = 'baseline'
    elif args.random:
        mode = 'random'
    else:
        mode = 'trained'
    
    run_gui_simulation(
        mode=mode,
        model_path=args.model,
        episodes=args.episodes
    )


if __name__ == "__main__":
    main()
