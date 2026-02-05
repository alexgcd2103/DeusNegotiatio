#!/usr/bin/env python3
"""
Oxford-Hyde Park Intersection - GUI Visualization

Run a trained DeusNegotiatio agent with SUMO-GUI visualization.
Includes high-fidelity metrics: queues, congestion, and time reduction.
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.oxford_hydepark_env import OxfordHydeParkEnv
from core.agent import DeusNegotiatioAgent

def run_gui_simulation(mode='trained', model_path='best_model.pth', episodes=1):
    """
    Run SUMO-GUI simulation with specified control mode and detailed metrics.
    """
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config/learning_hyperparams.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Create environment with GUI and custom view settings
    view_path = os.path.join(os.path.dirname(__file__), "network/view.xml")
    extra_args = ["--gui-settings-file", view_path]
    env = OxfordHydeParkEnv(use_gui=True, extra_sumo_args=extra_args)
    
    print(f"\n{'='*60}")
    print(f"  Oxford-Hyde Park Intersection - High Fidelity Visualization")
    print(f"  Mode: {mode.upper()} | Device: {device}")
    print(f"  Projected Layout: LOS C Modified (2046)")
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
        
        full_model_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_model_path):
            agent.load(full_model_path)
            agent.epsilon = 0.0
            print(f"  Loaded trained model: {model_path}")
        else:
            print(f"  WARNING: Model {model_path} not found. Using random actions.")
            mode = 'random'
    
    # Track metrics
    initial_delay = None

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        
        try:
            while True:
                if mode == 'trained' and agent is not None:
                    action = agent.select_action(state)
                elif mode == 'baseline':
                    action = 0 
                else:
                    action = env.action_space.sample()
                
                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
                step += 1
                
                # --- Metrics Calculation ---
                total_wait = info.get('total_wait_time', 0)
                veh_count = info.get('vehicles_in_sim', 0)
                avg_delay = total_wait / max(1, veh_count)
                
                # Congestion: Ratio of halting vehicles to total vehicles
                congestion = (info.get('queue_length', 0) / max(1, veh_count)) * 100
                
                if initial_delay is None and step > 20: 
                    initial_delay = avg_delay
                
                reduction = initial_delay - avg_delay if initial_delay else 0
                
                # Real-time dashboard
                if step % 2 == 0:
                    print(f"  [Step {step:4d}] "
                          f"Queues: {info.get('queue_length', 0):3d} | "
                          f"Congestion: {congestion:4.1f}% | "
                          f"Avg Delay: {avg_delay:6.1f}s | "
                          f"AI Improvement: {reduction:+6.1f}s", end='\r')
                
                if terminated or truncated:
                    break
                    
        except KeyboardInterrupt:
            print("\n  Simulation interrupted by user.")
            break
            
        print(f"\n\n  Episode Result:")
        print(f"  - Avg Congestion: {congestion:.1f}%")
        print(f"  - Final Delay Savings: {reduction:.1f} seconds/vehicle")
        print(f"  - Total Vehicles Processed: {info.get('throughput', 0)}")

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--model', type=str, default='best_model.pth')
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()
    
    mode = 'baseline' if args.baseline else ('random' if args.random else 'trained')
    run_gui_simulation(mode=mode, model_path=args.model, episodes=args.episodes)

if __name__ == "__main__":
    main()
