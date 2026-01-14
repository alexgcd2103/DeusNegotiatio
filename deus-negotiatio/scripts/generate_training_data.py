import sys
import os
import multiprocessing as mp
import numpy as np
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oxford_hydepark_env import OxfordHydeParkEnv

def run_single_episode(episode_config):
    """Run one simulation episode and collect data"""
    episode_id, scenario, seed = episode_config
    
    np.random.seed(seed)
    
    # Note: route_file path resolution might need adjustment depending on where this is run
    # For now assuming running from project root or routes are available
    env = OxfordHydeParkEnv(
        route_file=f'routes_{scenario}.rou.xml',
        num_seconds=3600,  # 1 hour episodes
        use_gui=False
    )
    
    try:
        state = env.reset()
    except Exception as e:
        # Fallback if SUMO/routes missing, just to show structure (or raising error)
        # For now, let it raise so we know setup is incomplete
        raise e

    episode_data = []
    total_reward = 0
    
    done = False
    while not done:
        # Random policy for data collection (exploratory)
        action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(action)
        
        # Store transition
        episode_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info
        })
        
        state = next_state
        total_reward += reward
    
    env.close()
    
    return {
        'episode_id': episode_id,
        'scenario': scenario,
        'transitions': episode_data,
        'total_reward': total_reward,
        'metrics': {
            'avg_wait_time': np.mean([t['info']['total_wait_time'] for t in episode_data]),
            'avg_queue': np.mean([t['info']['queue_length'] for t in episode_data]),
            'throughput': sum([t['info']['throughput'] for t in episode_data])
        }
    }

def generate_massive_dataset(num_episodes=50000, num_workers=16):
    """Generate large-scale training dataset using parallel workers"""
    
    scenarios = [
        'am_rush', 'pm_rush', 'midday', 'evening', 'night',
        'special_event', 'accident', 'weather'
    ] * (num_episodes // 8 + 1)  # Repeat scenarios to reach target episodes
    
    # Create episode configurations
    episode_configs = [
        (i, scenarios[i], np.random.randint(0, 1000000))
        for i in range(num_episodes)
    ]
    
    # Create output directory
    os.makedirs('training_data', exist_ok=True)
    
    # Parallel execution
    print(f"Generating {num_episodes} episodes using {num_workers} workers...")
    
    # If using mp.Pool, ensure protecting entry point
    # But since this function is called, it should be fine if main is protected.
    
    # Note: For multiprocessing with SUMO, each worker needs its own port or Careful management
    # standard sumo-rl handles this via unique port assignment usually, or try/retry.
    # Here assuming standard execution.
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(run_single_episode, episode_configs),
            total=num_episodes
        ))
    
    # Save dataset in chunks (avoid memory overflow)
    chunk_size = 5000
    for i in range(0, len(results), chunk_size):
        chunk = results[i:i+chunk_size]
        with open(f'training_data/dataset_chunk_{i//chunk_size:04d}.pkl', 'wb') as f:
            pickle.dump(chunk, f)
    
    print(f"Dataset generation complete. Saved {len(results)} episodes.")
    
    return results

if __name__ == '__main__':
    # Generate 50,000 episodes
    # Configurable
    generate_massive_dataset(num_episodes=10, num_workers=2) # Default small for testing
