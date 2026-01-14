# SUMO Multi-Agent Integration Guide

**Project**: Oxford–Hyde Park Intersection RL Training  
**Framework**: SUMO 1.25.0 + Multi-Agent RL (PPO/DQN)  
**Environment**: macOS/Linux with Anaconda  

This guide covers **full end-to-end SUMO setup** and how to integrate it with a multi-agent RL system.

---

## 1. Prerequisites & Environment Setup

### 1.1 System Requirements

**macOS (your setup)**:
```bash
# Verify SUMO is installed
which sumo
# Should output: /Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO/bin/sumo

# Check SUMO version
sumo --version
# Should output: Version 1.25.0
```

**Linux (alternative)**:
```bash
# Install SUMO via package manager
sudo apt-get install sumo sumo-doc sumo-tools
# Or compile from source: https://sumo.dlr.de/docs/Installing/index.html
```

### 1.2 Python Environment

**Create Anaconda environment**:
```bash
conda create -n fairenv python=3.10
conda activate fairenv

# Core dependencies
pip install numpy pandas scipy matplotlib seaborn

# SUMO tools
pip install sumolib lxml xmltodict

# RL frameworks (choose one or both)
pip install stable-baselines3[extra]  # PPO, DQN, etc.
pip install torch torchvision torchaudio  # For PyTorch

# Gym/environment
pip install gymnasium gym

# Utilities
pip install tensorboard tqdm pyyaml configparser
```

**Verify SUMO Python bindings**:
```python
python -c "import sys; sys.path.insert(0, '/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO/tools'); import traci; print('✓ TraCI imported successfully')"
```

### 1.3 Project Directory Structure

```
DeusNegatiatio/
├── network/
│   ├── oxford_hydepark.net.xml          # Network geometry (SUMO)
│   ├── oxford_hydepark.rou.xml          # Routes & demand
│   ├── oxford_hydepark.sumocfg          # SUMO config
│   ├── tls.add.xml                      # Traffic light definitions
│   └── detectors.xml                    # Induction loops (optional)
│
├── envs/
│   ├── __init__.py
│   ├── oxford_hydepark_env.py           # Single-agent wrapper
│   ├── multi_agent_env.py               # Multi-agent wrapper
│   └── sumo_utils.py                    # Helper functions
│
├── agents/
│   ├── __init__.py
│   ├── traffic_light_agent.py           # Individual agent class
│   ├── coordinator.py                   # Multi-agent coordinator
│   └── reward_shaper.py                 # Reward calculation
│
├── training/
│   ├── train_single.py                  # Single-agent training
│   ├── train_multi.py                   # Multi-agent training
│   ├── config.yaml                      # Hyperparameters
│   └── callbacks.py                     # Custom callbacks
│
├── evaluation/
│   ├── evaluate.py                      # Run trained policy
│   ├── metrics.py                       # Analysis tools
│   └── visualize.py                     # Generate plots
│
├── data/
│   ├── logs/                            # TensorBoard logs
│   ├── models/                          # Saved policies
│   └── results/                         # Episode statistics
│
├── main.py                              # Entry point
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## 2. SUMO Network Files (Complete Setup)

### 2.1 Network Geometry (`oxford_hydepark.net.xml`)

**Create using netedit**:
```bash
# Launch netedit GUI
netedit
# File → New Network
# Create 4-leg junction manually, or:
# Edit → OpenStreetMap → Download area → Import
```

**Or use netconvert (command-line)**:
```bash
# From OpenDRIVE or OSM file
netconvert --opendrive input.xodr -o oxford_hydepark.net.xml

# From plain XML (edges/nodes)
netconvert --edge-files edges.xml --node-files nodes.xml -o oxford_hydepark.net.xml
```

**Minimal hand-coded example** (`oxford_hydepark.net.xml`):
```xml
<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <!-- Nodes (intersection corners) -->
    <node id="south" x="0" y="-200" type="priority"/>
    <node id="north" x="0" y="200" type="priority"/>
    <node id="west" x="-200" y="0" type="priority"/>
    <node id="east" x="200" y="0" type="priority"/>
    <node id="center" x="0" y="0" type="traffic_light"/>
    
    <!-- Edges (road segments) -->
    <!-- Northbound approach (South → Center) -->
    <edge id="north_in" from="south" to="center" priority="2" numLanes="4" speed="16.67" length="115">
        <lane id="north_in_0" index="0" speed="16.67" length="115" width="3.0"/>
        <lane id="north_in_1" index="1" speed="16.67" length="115" width="3.3"/>
        <lane id="north_in_2" index="2" speed="16.67" length="115" width="3.3"/>
        <lane id="north_in_3" index="3" speed="16.67" length="115" width="3.0"/>
    </edge>
    
    <!-- Northbound departure (Center → North) -->
    <edge id="north_out" from="center" to="north" priority="2" numLanes="4" speed="16.67" length="50">
        <lane id="north_out_0" index="0" speed="16.67" length="50" width="3.0"/>
        <lane id="north_out_1" index="1" speed="16.67" length="50" width="3.3"/>
        <lane id="north_out_2" index="2" speed="16.67" length="50" width="3.3"/>
        <lane id="north_out_3" index="3" speed="16.67" length="50" width="3.0"/>
    </edge>
    
    <!-- Repeat for South, East, West approaches... -->
    <!-- (abbreviated for brevity) -->
    
    <!-- Internal connections (manage turns at junction) -->
    <connection from="north_in" to="north_out" fromLane="1" toLane="1"/>
    <connection from="north_in" to="north_out" fromLane="2" toLane="2"/>
    <connection from="north_in" to="east_out" fromLane="0" toLane="0"/>
    <connection from="north_in" to="west_out" fromLane="3" toLane="3"/>
    <!-- ... more connections ... -->
    
    <!-- Traffic Light Logic -->
    <tlLogic id="center" type="static" programID="0" offset="0">
        <phase duration="40" state="GGrrGGrrrrrrrrr" name="NS_through"/>
        <phase duration="4"  state="yyrryyrrrrrrrr" name="NS_yellow"/>
        <phase duration="20" state="rrrGrrrrrrrGrr" name="NS_left"/>
        <phase duration="4"  state="rrryrrrrrrryrr" name="NS_left_yellow"/>
        <phase duration="40" state="rrGGrrrrGGrrrr" name="EW_through"/>
        <phase duration="4"  state="rryyrrrryyrrrr" name="EW_yellow"/>
        <phase duration="20" state="rrrrrrrGrrrrrG" name="EW_left"/>
        <phase duration="4"  state="rrrrrrryrrrrrry" name="EW_left_yellow"/>
    </tlLogic>
</net>
```

**Validate network**:
```bash
# Check for errors
sumo --net-file oxford_hydepark.net.xml --check-only

# Visualize in sumo-gui
sumo-gui -c oxford_hydepark.sumocfg
```

### 2.2 Traffic Demand (`oxford_hydepark.rou.xml`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="16.67"/>
    <vType id="truck" accel="1.3" decel="4.0" sigma="0.5" length="12.0" maxSpeed="13.89"/>
    
    <!-- Routes (movement patterns) -->
    <route id="NB_L" edges="north_in west_out"/>
    <route id="NB_T" edges="north_in north_out"/>
    <route id="NB_R" edges="north_in east_out"/>
    <route id="EB_L" edges="east_in north_out"/>
    <route id="EB_T" edges="east_in east_out"/>
    <route id="EB_R" edges="east_in south_out"/>
    <!-- ... continue for all 12 movements ... -->
    
    <!-- Flows (2046 PM DHV) -->
    <flow id="NB_L_cars" type="car" route="NB_L" begin="0" end="3600" number="74"/>
    <flow id="NB_T_cars" type="car" route="NB_T" begin="0" end="3600" number="642"/>
    <flow id="NB_R_cars" type="car" route="NB_R" begin="0" end="3600" number="68"/>
    <flow id="NB_trucks" type="truck" route="NB_T" begin="0" end="3600" number="11"/>
    
    <flow id="EB_L_cars" type="car" route="EB_L" begin="0" end="3600" number="529"/>
    <flow id="EB_T_cars" type="car" route="EB_T" begin="0" end="3600" number="897"/>
    <flow id="EB_R_cars" type="car" route="EB_R" begin="0" end="3600" number="53"/>
    <flow id="EB_trucks" type="truck" route="EB_T" begin="0" end="3600" number="6"/>
    
    <flow id="SB_L_cars" type="car" route="SB_L" begin="0" end="3600" number="468"/>
    <flow id="SB_T_cars" type="car" route="SB_T" begin="0" end="3600" number="767"/>
    <flow id="SB_R_cars" type="car" route="SB_R" begin="0" end="3600" number="731"/>
    <flow id="SB_trucks" type="truck" route="SB_T" begin="0" end="3600" number="15"/>
    
    <flow id="WB_L_cars" type="car" route="WB_L" begin="0" end="3600" number="199"/>
    <flow id="WB_T_cars" type="car" route="WB_T" begin="0" end="3600" number="829"/>
    <flow id="WB_R_cars" type="car" route="WB_R" begin="0" end="3600" number="441"/>
    <flow id="WB_trucks" type="truck" route="WB_T" begin="0" end="3600" number="14"/>
    
    <!-- Pedestrians -->
    <personFlow id="ped_NB" begin="0" end="3600" number="186">
        <walk edges="NB_ped_crossing"/>
    </personFlow>
    <personFlow id="ped_EB" begin="0" end="3600" number="94">
        <walk edges="EB_ped_crossing"/>
    </personFlow>
</routes>
```

### 2.3 SUMO Configuration (`oxford_hydepark.sumocfg`)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network/oxford_hydepark.net.xml"/>
        <route-files value="network/oxford_hydepark.rou.xml"/>
        <additional-files value="network/detectors.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
    
    <output>
        <summary-output value="data/output/summary.xml"/>
        <queue-output value="data/output/queue.xml"/>
        <lanechange-output value="data/output/lanechange.xml"/>
    </output>
    
    <processing>
        <time-to-teleport value="300"/>
        <no-warnings value="true"/>
        <precision value="2"/>
    </processing>
    
    <gui>
        <gui-settings-file value="gui-settings.xml"/>
        <start value="false"/>
        <delay value="0"/>
    </gui>
</configuration>
```

### 2.4 Induction Loops (Optional, for detector data)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <!-- Queue detectors at stop line (12 total = 4 approaches × 3 lanes) -->
    
    <!-- Northbound -->
    <inductionLoop id="det_NB_0" lane="north_in_0" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_NB_1" lane="north_in_1" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_NB_2" lane="north_in_2" pos="10" period="1" file="data/output/detectors.xml"/>
    
    <!-- Eastbound -->
    <inductionLoop id="det_EB_0" lane="east_in_0" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_EB_1" lane="east_in_1" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_EB_2" lane="east_in_2" pos="10" period="1" file="data/output/detectors.xml"/>
    
    <!-- Southbound -->
    <inductionLoop id="det_SB_0" lane="south_in_0" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_SB_1" lane="south_in_1" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_SB_2" lane="south_in_2" pos="10" period="1" file="data/output/detectors.xml"/>
    
    <!-- Westbound -->
    <inductionLoop id="det_WB_0" lane="west_in_0" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_WB_1" lane="west_in_1" pos="10" period="1" file="data/output/detectors.xml"/>
    <inductionLoop id="det_WB_2" lane="west_in_2" pos="10" period="1" file="data/output/detectors.xml"/>
</additional>
```

---

## 3. Python Environment Wrapper

### 3.1 SUMO Connection Manager (`envs/sumo_utils.py`)

```python
import os
import sys
import subprocess
import time
import random
from pathlib import Path

# Set SUMO_HOME
SUMO_HOME = '/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO'
os.environ['SUMO_HOME'] = SUMO_HOME
sys.path.insert(0, os.path.join(SUMO_HOME, 'tools'))

import traci
import sumolib

class SUMOSimulator:
    """Manages SUMO process and TraCI connection."""
    
    def __init__(self, sumocfg_path, gui=False, port=None, label='sim'):
        self.sumocfg_path = sumocfg_path
        self.gui = gui
        self.port = port or random.randint(8000, 9000)
        self.label = label
        self.sumo_process = None
        self.is_running = False
    
    def start(self):
        """Launch SUMO and connect TraCI."""
        try:
            # Close any existing connection
            if self.is_running:
                self.close()
            
            time.sleep(0.1)
            
            # Build SUMO command
            binary = sumolib.checkBinary('sumo-gui' if self.gui else 'sumo')
            cmd = [
                binary,
                '-c', self.sumocfg_path,
                '--no-warnings',
                '--start',  # Auto-start simulation
            ]
            
            # Start TraCI connection
            traci.start(
                cmd,
                port=self.port,
                numRetries=10,
                label=self.label
            )
            
            self.is_running = True
            print(f"✓ SUMO started on port {self.port}")
            
        except Exception as e:
            print(f"✗ Failed to start SUMO: {e}")
            raise
    
    def step(self, action=None):
        """Execute one simulation step."""
        if not self.is_running:
            raise RuntimeError("SUMO not running")
        
        if action is not None:
            # Apply action (e.g., change traffic light phase)
            pass
        
        traci.simulationStep()
    
    def close(self):
        """Close SUMO and TraCI connection."""
        try:
            if self.is_running:
                traci.close(False)
                self.is_running = False
                print(f"✓ SUMO closed (port {self.port})")
        except Exception as e:
            print(f"⚠ Error closing SUMO: {e}")
    
    def __del__(self):
        self.close()
```

### 3.2 Single-Agent Environment (`envs/oxford_hydepark_env.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.sumo_utils import SUMOSimulator
import traci

class OxfordHydeParkEnv(gym.Env):
    """Gym environment for Oxford-Hyde Park intersection."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, sumocfg_path, gui=False, seed=None):
        super().__init__()
        
        self.sumocfg_path = sumocfg_path
        self.gui = gui
        self.seed_value = seed
        
        self.sim = SUMOSimulator(sumocfg_path, gui=gui)
        self.sim.start()
        
        # Observation space: queue lengths + phase info
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(30,), dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        # 0-3: Switch to phase 1-4
        # 4: Extend current phase
        self.action_space = spaces.Discrete(5)
        
        self.current_phase = 0
        self.phase_duration = 0
        self.step_count = 0
        self.episode_length = 3600  # 1 hour
        
        # TLS control
        self.tls_id = 'center'
        self.num_phases = 8
    
    def reset(self, seed=None):
        """Reset environment for new episode."""
        self.sim.close()
        time.sleep(0.5)
        self.sim = SUMOSimulator(self.sumocfg_path, gui=self.gui)
        self.sim.start()
        
        self.current_phase = 0
        self.phase_duration = 0
        self.step_count = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute action and return transition."""
        # Apply traffic light action
        if action < 4:
            # Switch to phase (0-3: phases 0, 2, 4, 6 in SUMO)
            target_phase = action * 2
            if target_phase != self.current_phase:
                traci.traffic_light.setPhase(self.tls_id, target_phase)
                self.current_phase = target_phase
                self.phase_duration = 0
        elif action == 4:
            # Extend current phase (do nothing)
            pass
        
        self.phase_duration += 1
        
        # Simulate one step
        self.sim.step()
        self.step_count += 1
        
        # Get observations and reward
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self.step_count >= self.episode_length
        
        return obs, reward, done, False, {}
    
    def _get_observation(self):
        """Collect state from SUMO."""
        obs = np.zeros(30, dtype=np.float32)
        
        # Queue lengths (12 lanes)
        lane_ids = [
            'north_in_0', 'north_in_1', 'north_in_2',
            'east_in_0', 'east_in_1', 'east_in_2',
            'south_in_0', 'south_in_1', 'south_in_2',
            'west_in_0', 'west_in_1', 'west_in_2',
        ]
        
        for i, lane_id in enumerate(lane_ids):
            try:
                queue_len = traci.lane.getLastStepHaltingNumber(lane_id)
                obs[i] = queue_len
            except:
                obs[i] = 0
        
        # Current phase (one-hot)
        obs[24:28] = 0
        obs[24 + (self.current_phase // 2)] = 1
        
        # Time since phase change
        obs[28] = min(self.phase_duration / 60.0, 1.0)
        
        # Pedestrian wait (simplified)
        obs[29] = 0  # TODO: implement
        
        return obs
    
    def _compute_reward(self):
        """Calculate reward based on traffic metrics."""
        # Queue penalty
        total_queue = 0
        lane_ids = [...]  # Same as above
        for lane_id in lane_ids:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                total_queue += queue
            except:
                pass
        
        # Waiting time penalty
        total_wait = traci.simulation.getScale()
        
        # Throughput reward
        vehicles_departed = traci.simulation.getDepartedNumber()
        
        # Combine
        reward = (
            -0.01 * total_wait
            - 0.05 * total_queue
            + 0.5 * vehicles_departed
        )
        
        return reward
    
    def close(self):
        self.sim.close()
```

### 3.3 Multi-Agent Environment (`envs/multi_agent_env.py`)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
from envs.sumo_utils import SUMOSimulator

class MultiAgentOxfordHydeParkEnv(gym.Env):
    """Multi-agent environment: one agent per traffic light phase group."""
    
    def __init__(self, sumocfg_path, num_agents=2, gui=False):
        """
        num_agents: 2 = {N-S agent, E-W agent}
                    or 4 = {NB agent, EB agent, SB agent, WB agent}
        """
        self.num_agents = num_agents
        self.sumocfg_path = sumocfg_path
        self.gui = gui
        
        self.sim = SUMOSimulator(sumocfg_path, gui=gui)
        self.sim.start()
        
        # Per-agent observation/action spaces
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # {Extend, Phase1, Phase2}
        
        self.agents = {f'agent_{i}': i for i in range(num_agents)}
        self.step_count = 0
        self.episode_length = 3600
    
    def reset(self, seed=None):
        self.sim.close()
        time.sleep(0.5)
        self.sim = SUMOSimulator(self.sumocfg_path, gui=self.gui)
        self.sim.start()
        
        self.step_count = 0
        obs = {agent_id: self._get_agent_obs(i) for agent_id, i in self.agents.items()}
        return obs, {}
    
    def step(self, actions):
        """
        actions: dict {agent_id: action_value}
        """
        # Synchronize agents' decisions into unified TLS phases
        if self.num_agents == 2:
            # Agent 0 controls N-S (phases 0, 2)
            # Agent 1 controls E-W (phases 4, 6)
            ns_action = actions.get('agent_0', 0)
            ew_action = actions.get('agent_1', 0)
            
            if ns_action == 1:
                traci.traffic_light.setPhase(self.tls_id, 0)
            elif ns_action == 2:
                traci.traffic_light.setPhase(self.tls_id, 2)
            
            if ew_action == 1:
                traci.traffic_light.setPhase(self.tls_id, 4)
            elif ew_action == 2:
                traci.traffic_light.setPhase(self.tls_id, 6)
        
        self.sim.step()
        self.step_count += 1
        
        # Observations, rewards, done flags
        obs = {agent_id: self._get_agent_obs(i) for agent_id, i in self.agents.items()}
        rewards = {agent_id: self._compute_agent_reward(i) for agent_id, i in self.agents.items()}
        done = self.step_count >= self.episode_length
        dones = {agent_id: done for agent_id in self.agents}
        dones['__all__'] = done
        
        return obs, rewards, dones, False, {}
    
    def _get_agent_obs(self, agent_id):
        """Get observation for specific agent."""
        obs = np.zeros(15, dtype=np.float32)
        
        if agent_id == 0:  # N-S agent
            lanes = ['north_in_0', 'north_in_1', 'north_in_2',
                     'south_in_0', 'south_in_1', 'south_in_2']
        else:  # E-W agent
            lanes = ['east_in_0', 'east_in_1', 'east_in_2',
                     'west_in_0', 'west_in_1', 'west_in_2']
        
        for i, lane_id in enumerate(lanes):
            try:
                obs[i] = traci.lane.getLastStepHaltingNumber(lane_id)
            except:
                obs[i] = 0
        
        return obs
    
    def _compute_agent_reward(self, agent_id):
        """Compute reward for agent."""
        # Aggregate metrics for agent's lanes
        if agent_id == 0:
            lanes = ['north_in_0', 'north_in_1', 'north_in_2',
                     'south_in_0', 'south_in_1', 'south_in_2']
        else:
            lanes = ['east_in_0', 'east_in_1', 'east_in_2',
                     'west_in_0', 'west_in_1', 'west_in_2']
        
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        reward = -0.05 * total_queue
        
        return reward
    
    def close(self):
        self.sim.close()
```

---

## 4. Multi-Agent Training Script

### 4.1 Configuration (`training/config.yaml`)

```yaml
# Training configuration
environment:
  sumocfg_path: "network/oxford_hydepark.sumocfg"
  gui: false
  num_agents: 2
  episode_length: 3600

training:
  algorithm: "PPO"  # or "DQN"
  total_timesteps: 1000000
  learning_rate: 3e-4
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95

evaluation:
  eval_frequency: 10000
  n_eval_episodes: 10
  save_best: true

logging:
  log_dir: "data/logs"
  tensorboard: true
```

### 4.2 Multi-Agent Training (`training/train_multi.py`)

```python
import os
import yaml
import time
import numpy as np
from pathlib import Path

# Set SUMO_HOME before imports
os.environ['SUMO_HOME'] = '/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO'

from envs.multi_agent_env import MultiAgentOxfordHydeParkEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
import torch

class TrainingConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.env_cfg = cfg['environment']
        self.train_cfg = cfg['training']
        self.eval_cfg = cfg['evaluation']
        self.log_cfg = cfg['logging']

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_dir):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.best_reward = -np.inf
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward = self._evaluate()
            
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(os.path.join(self.log_dir, 'best_model'))
                print(f"✓ New best reward: {mean_reward:.2f}")
        
        return True
    
    def _evaluate(self, n_episodes=10):
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Evaluate all agents
                actions = {}
                for agent_id in self.eval_env.agents:
                    action, _ = self.model.predict(obs[agent_id], deterministic=True)
                    actions[agent_id] = action
                
                obs, rewards_dict, dones, _, _ = self.eval_env.step(actions)
                episode_reward += sum(rewards_dict.values())
                done = dones['__all__']
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)

def train_multi_agent():
    print("=" * 60)
    print("SUMO Multi-Agent RL Training")
    print("=" * 60)
    
    # Load config
    config = TrainingConfig("training/config.yaml")
    
    # Create environments
    print("\n[1/5] Initializing SUMO environments...")
    env = MultiAgentOxfordHydeParkEnv(
        config.env_cfg['sumocfg_path'],
        num_agents=config.env_cfg['num_agents'],
        gui=config.env_cfg['gui']
    )
    eval_env = MultiAgentOxfordHydeParkEnv(
        config.env_cfg['sumocfg_path'],
        num_agents=config.env_cfg['num_agents'],
        gui=False
    )
    print(f"✓ Created {config.env_cfg['num_agents']}-agent environment")
    
    # Create log directory
    log_dir = Path(config.log_cfg['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("\n[2/5] Initializing PPO policy...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.train_cfg['learning_rate'],
        batch_size=config.train_cfg['batch_size'],
        n_epochs=config.train_cfg['n_epochs'],
        gamma=config.train_cfg['gamma'],
        gae_lambda=config.train_cfg['gae_lambda'],
        tensorboard_log=str(log_dir),
        device=device,
        verbose=1
    )
    
    # Callback
    callback = CustomCallback(
        eval_env,
        eval_freq=config.eval_cfg['eval_frequency'],
        log_dir=str(log_dir)
    )
    
    # Train
    print("\n[3/5] Starting training...")
    print(f"   Total timesteps: {config.train_cfg['total_timesteps']:,}")
    
    start_time = time.time()
    model.learn(
        total_timesteps=config.train_cfg['total_timesteps'],
        callback=callback,
        progress_bar=True
    )
    elapsed = time.time() - start_time
    
    print(f"\n[4/5] Training complete ({elapsed/60:.1f} minutes)")
    
    # Save final model
    model.save(os.path.join(log_dir, 'final_model'))
    print("[5/5] Model saved")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {log_dir}")
    print("=" * 60)

if __name__ == '__main__':
    train_multi_agent()
```

---

## 5. Running the Simulation

### 5.1 Quick Start

```bash
# Activate environment
conda activate fairenv

# Set SUMO_HOME
export SUMO_HOME="/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO"

# Run training
cd DeusNegotiatio
python training/train_multi.py

# Monitor progress
tensorboard --logdir data/logs
# Open http://localhost:6006
```

### 5.2 Troubleshooting

**SUMO won't start**:
```bash
# Verify SUMO executable
ls -la $SUMO_HOME/bin/sumo

# Test TraCI connection
python -c "
import sys
sys.path.insert(0, '/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO/tools')
import traci; print('✓ TraCI OK')
"
```

**Port conflicts**:
```python
# In train_multi.py, ensure random port assignment
port = random.randint(8000, 9000)
```

**GPU not detected**:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## 6. Evaluation & Metrics

### 6.1 Evaluation Script (`evaluation/evaluate.py`)

```python
import os
os.environ['SUMO_HOME'] = '/Library/Frameworks/EclipseSUMO.framework/Versions/1.25.0/EclipseSUMO'

from envs.multi_agent_env import MultiAgentOxfordHydeParkEnv
from stable_baselines3 import PPO
import numpy as np

def evaluate_policy(model_path, num_episodes=10, gui=True):
    """Run trained policy and collect metrics."""
    
    env = MultiAgentOxfordHydeParkEnv(
        "network/oxford_hydepark.sumocfg",
        num_agents=2,
        gui=gui
    )
    
    model = PPO.load(model_path)
    
    metrics = {
        'total_rewards': [],
        'avg_queue': [],
        'max_queue': []
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        queues = []
        
        while not done:
            actions = {}
            for agent_id in env.agents:
                action, _ = model.predict(obs[agent_id], deterministic=True)
                actions[agent_id] = action
            
            obs, rewards, dones, _, _ = env.step(actions)
            ep_reward += sum(rewards.values())
            
            # Collect queue data
            import traci
            lane_queues = [traci.lane.getLastStepHaltingNumber(f'lane_{i}') 
                          for i in range(12)]
            queues.append(np.mean(lane_queues))
            
            done = dones['__all__']
        
        metrics['total_rewards'].append(ep_reward)
        metrics['avg_queue'].append(np.mean(queues))
        metrics['max_queue'].append(np.max(queues))
    
    env.close()
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Episode Reward: {np.mean(metrics['total_rewards']):.2f}")
    print(f"Std Episode Reward: {np.std(metrics['total_rewards']):.2f}")
    print(f"Avg Queue Length: {np.mean(metrics['avg_queue']):.1f} m")
    print(f"Max Queue Length: {np.mean(metrics['max_queue']):.1f} m")
    print("="*60 + "\n")
    
    return metrics

if __name__ == '__main__':
    metrics = evaluate_policy('data/logs/best_model.zip', num_episodes=10, gui=False)
```

---

## 7. Deployment Checklist

- [ ] SUMO installed and `SUMO_HOME` set
- [ ] Python environment with requirements installed
- [ ] Network files (.net.xml, .rou.xml, .sumocfg) in `network/` directory
- [ ] TraCI connection tested independently
- [ ] Multi-agent environment wrapper compiles without errors
- [ ] Training config (config.yaml) reviewed and tuned
- [ ] Training can run for at least 1 full episode without crashing
- [ ] TensorBoard logs generated and viewable
- [ ] Evaluation script can load and run trained model
- [ ] Metrics match engineering targets (queue < 100 m, LOS C, etc.)

---

## 8. Additional Resources

- **SUMO Documentation**: https://sumo.dlr.de/docs/
- **TraCI Python**: https://sumo.dlr.de/docs/TraCI/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gym/Gymnasium**: https://gymnasium.farama.org/

---

**End of Multi-Agent Integration Guide**
