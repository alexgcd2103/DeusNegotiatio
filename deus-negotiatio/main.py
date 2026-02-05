import argparse
import yaml
import torch
import sys
import os
import random
import warnings

# Suppress known SUMO warning that occurs despite correct config
warnings.filterwarnings("ignore", message="Environment variable SUMO_HOME is not set properly")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent import DeusNegotiatioAgent
from learning.training_loop import TrainingLoop
# Mock Environment for testing
class MockEnv:
    def __init__(self, state_dim=32, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def reset(self):
        return [0.5] * self.state_dim, {}
        
    def step(self, action):
        # Create a dynamic state for testing
        next_state = [random.random() for _ in range(self.state_dim)]
        # Reward varies based on action to simulate "learning" opportunity
        # e.g., action 0 is "correct" for some states
        reward = random.random() * 10
        terminated = False
        truncated = False
        return next_state, reward, terminated, truncated, {}


def ensure_sumo_home():
    """Auto-detect and set SUMO_HOME if not already set"""
    if 'SUMO_HOME' in os.environ:
        return

    # Known paths on Mac/Linux
    possible_paths = [
        "/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO",  # Found on user's system
        "/opt/homebrew/share/sumo",
        "/usr/local/share/sumo",
        "/usr/share/sumo"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Auto-detected SUMO_HOME: {path}")
            os.environ['SUMO_HOME'] = path
            
            # Ensure bin is in PATH so traci can find 'sumo' or 'sumo-gui'
            bin_path = os.path.join(path, 'bin')
            if bin_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')

            # Ensure tools are in python path
            tools_path = os.path.join(path, 'tools')
            if tools_path not in sys.path:
                sys.path.append(tools_path)
            return
            
    print("Warning: SUMO_HOME not detected. SUMO integration may fail.")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    ensure_sumo_home()
    
    parser = argparse.ArgumentParser(description="DeusNegotiatio Main")
    parser.add_argument('--config', type=str, default='config/learning_hyperparams.yaml', help='Path to config file')
    parser.add_argument('--test', action='store_true', help='Run self-test')
    parser.add_argument('--episodes', type=int, help='Override number of episodes')
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    
    # Create simple default config if loading fails or for testing
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using defaults.")
        config = {'learning_rate': 1e-4, 'batch_size': 32}

    if args.episodes:
        config['num_episodes'] = args.episodes
        
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    
    if args.test:
        print("Running System Initialization Test...")
        # Create mock agent for testing
        test_agent = DeusNegotiatioAgent(agent_id="Test_Agent", config=config, state_dim=32, action_dim=4, device=device)
        assert test_agent.policy_net is not None
        print("Agent initialized successfully.")
        return

    # Normal Mode
    print("Initializing Training Loop...")
    
    # Initialize Environment
    try:
         from envs.oxford_hydepark_env import OxfordHydeParkEnv
         env = OxfordHydeParkEnv()
         print("Initialized OxfordHydeParkEnv (SUMO)")
         state_dim = env.observation_space.shape[0]
         action_dim = env.action_space.n
    except Exception as e:
         print(f"Failed to initialize SUMO environment: {e}")
         print("Falling back to MockEnv")
         env = MockEnv()
         state_dim = 32
         action_dim = 4

    # Initialize Agent with correct dims
    agent = DeusNegotiatioAgent(
        agent_id="Int_1", 
        config=config, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device
    )

    trainer = TrainingLoop(agent, env, config)
    trainer.run()

if __name__ == "__main__":
    main()
