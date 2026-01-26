"""
Oxford-Hyde Park Intersection Environment (Multi-Sensor)

This environment connects to SUMO via TraCI to train an RL agent
for adaptive traffic signal control with multi-sensor fusion.

Network: 2046 PM Peak (6,371 veh/hr)
Sensors: LiDAR, Infrared, Motion Detection, Environmental Effects
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import time
import os

from sensors.lidar_simulator import LidarSimulator
from sensors.infrared_simulator import InfraredSimulator
from sensors.motion_detector import MotionDetector
from sensors.environmental_effects import EnvironmentalEffects


class OxfordHydeParkEnv(gym.Env):
    """
    Multi-Sensor Gym Environment for Oxford-Hyde Park intersection.
    
    Observation Space: 156 features
        - SUMO ground truth (42 lane features + 6 phase encoding = 48)
        - LiDAR occupancy grid (8x8 = 64)
        - Infrared thermal (5)
        - Motion vectors (7)
        - Weather state (4)
        - Time features (2)
        - Padding (26)
    
    Action Space: 4 discrete actions
        - 0: NS Through Green
        - 1: NS Yellow -> EW Green
        - 2: EW Through Green  
        - 3: EW Yellow -> NS Green
    """
    
    def __init__(self, 
                 net_file='network/oxford_hyde_park.net.xml',
                 route_file='network/oxford_hyde_park.rou.xml',
                 add_file='network/oxford_hyde_park.add.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5,
                 yellow_time=4,
                 min_green=10,
                 max_green=55):
        
        super().__init__()
        
        # Resolve paths relative to deus-negotiatio directory
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.net_file = os.path.join(base_path, net_file)
        self.route_file = os.path.join(base_path, route_file)
        self.add_file = os.path.join(base_path, add_file)
        
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        
        # Traffic light ID from generated network
        self.ts_id = 'C'

        # Lane configurations matching generated network
        # NB/SB: 4 lanes each (0=Left, 1-2=Through, 3=Right)
        # EB/WB: 3 lanes each (0=Left, 1=Through, 2=Right)
        self.lanes = [
            # Northbound (4 lanes)
            'NB_in_0', 'NB_in_1', 'NB_in_2', 'NB_in_3',
            # Southbound (4 lanes)
            'SB_in_0', 'SB_in_1', 'SB_in_2', 'SB_in_3',
            # Eastbound (3 lanes)
            'EB_in_0', 'EB_in_1', 'EB_in_2',
            # Westbound (3 lanes)
            'WB_in_0', 'WB_in_1', 'WB_in_2'
        ]
        self.num_lanes = 14

        # --- Sensor Simulators ---
        self.lidar = LidarSimulator()
        self.infrared = InfraredSimulator()
        self.motion = MotionDetector()
        self.weather = EnvironmentalEffects()
        
        # Observation space (156-dim multi-sensor)
        # Lane features: 14 lanes × 3 features = 42
        # Phase encoding: 6 phases = 6
        # LiDAR grid: 8×8 = 64
        # Infrared: 5
        # Motion: 7
        # Weather: 4
        # Time: 2
        # Padding to 156: 26
        self.obs_dim = 156
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.obs_dim,),  
            dtype=np.float32
        )
        
        # 4 actions for main phase control
        # (yellows and all-reds are handled automatically by SUMO)
        self.action_space = spaces.Discrete(4)
        
        self.sumo_running = False
        self.simulation_step = 0
        self.last_action = 0
        
    def reset(self, seed=None, options=None):
        """Reset simulation to initial state"""
        super().reset(seed=seed)
        
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            time.sleep(0.5) 
        
        # Use full path to SUMO binary from SUMO_HOME
        sumo_home = os.environ.get('SUMO_HOME', '')
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        if sumo_home:
            sumo_binary = os.path.join(sumo_home, 'bin', sumo_binary)
        
        sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--additional-files', self.add_file,
            '--no-warnings',
            '--no-step-log',
            '--time-to-teleport', '-1',
            '--waiting-time-memory', '1000',
            '--max-depart-delay', '0',
            '--start'
        ]
        
        traci.start(sumo_cmd)
        self.sumo_running = True
        self.simulation_step = 0
        self.last_action = 0
        
        # Reset weather
        self.weather = EnvironmentalEffects()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, terminated, truncated, info"""
        
        # Apply action (set traffic light phase)
        # Actions map to main green phases only
        target_phase = self._action_to_phase(action)
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        
        # Only change phase if different from current main phase
        if target_phase != current_phase:
            traci.trafficlight.setPhase(self.ts_id, target_phase)
        
        # Simulate for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1
            self.weather.step()
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check if episode is done
        terminated = self.simulation_step >= self.num_seconds
        truncated = False
        
        info = {
            'total_wait_time': self._get_total_wait_time(),
            'queue_length': self._get_total_queue_length(),
            'throughput': self._get_throughput(),
            'vehicles_in_sim': len(traci.vehicle.getIDList())
        }
        
        self.last_action = action
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Construct multi-sensor state vector"""
        state = []
        
        # Collect raw vehicle data for sensors
        all_vehicles = []
        for veh_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(veh_id)
            all_vehicles.append({
                'id': veh_id,
                'x': x,
                'y': y,
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id),
                'angle': np.radians(traci.vehicle.getAngle(veh_id)),
                'length': traci.vehicle.getLength(veh_id)
            })
            
        # Center coordinates around (0,0) intersection
        # The network is centered at (0,0) so just use raw coordinates
        # but clamp to sensor range
        for v in all_vehicles:
            v['x'] = np.clip(v['x'], -100, 100)
            v['y'] = np.clip(v['y'], -100, 100)
        
        # --- 1. Base SUMO Features (42 = 14 lanes × 3 features) ---
        for lane in self.lanes:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                avg_speed = traci.lane.getLastStepMeanSpeed(lane)
                occupancy = traci.lane.getLastStepOccupancy(lane)
            except:
                queue, avg_speed, occupancy = 0, 0, 0
            # Normalize
            state.extend([
                queue / 50.0,           # Normalize queue (max ~50 vehicles)
                avg_speed / 16.67,      # Normalize speed (max 60 km/h = 16.67 m/s)
                occupancy / 100.0       # Occupancy is 0-100%
            ])
        
        # --- 2. Phase Encoding (6) ---
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        phase_encoding = np.zeros(6)
        if 0 <= current_phase < 6:
            phase_encoding[current_phase] = 1.0
        state.extend(phase_encoding)
        
        # --- 3. LiDAR Features (64 = 8×8 grid) ---
        lidar_data = self.lidar.scan(all_vehicles)
        lidar_data = self.weather.apply_to_lidar(lidar_data)
        state.extend(lidar_data['occupancy_grid'].flatten())
        
        # --- 4. Infrared Features (5) ---
        ir_data = self.infrared.capture(all_vehicles)
        state.extend(ir_data['quadrant_heat'])
        state.append(ir_data['emergency_heat'])
        
        # --- 5. Motion Features (7) ---
        motion_data = self.motion.detect(all_vehicles)
        motion_data = self.weather.apply_to_motion(motion_data)
        state.extend(motion_data['motion_vectors'])  # 4
        state.extend(motion_data['class_dist'])      # 3
        
        # --- 6. Weather Features (4) ---
        state.extend(self.weather.get_state_feature())
        
        # --- 7. Time Features (2) ---
        sim_time = traci.simulation.getTime()
        # Encode as position on 3600-second cycle (1 hour)
        progress = (sim_time % 3600) / 3600.0 * 2 * np.pi
        state.append(np.sin(progress))
        state.append(np.cos(progress))
        
        # Pad to obs_dim
        current_len = len(state)
        if current_len < self.obs_dim:
            state.extend([0.0] * (self.obs_dim - current_len))
        elif current_len > self.obs_dim:
            state = state[:self.obs_dim]
            
        return np.array(state, dtype=np.float32)
    
    def _compute_reward(self, action):
        """
        Multi-objective reward function optimized for 2046 PM Peak.
        
        Components:
        - Wait time penalty (dominant)
        - Queue length penalty
        - Throughput bonus
        - Phase change penalty (smooth transitions)
        - Westbound priority (highest volume approach)
        """
        total_wait_time = self._get_total_wait_time()
        total_queue = self._get_total_queue_length()
        throughput = self._get_throughput()
        
        # Westbound queue (priority - 41% of total volume)
        wb_queue = 0
        for lane in ['WB_in_0', 'WB_in_1', 'WB_in_2']:
            try:
                wb_queue += traci.lane.getLastStepHaltingNumber(lane)
            except:
                pass
        
        # Phase change penalty
        phase_change_penalty = -0.5 if action != self.last_action else 0.0
        
        # Weighted reward
        reward = (
            -0.3 * total_wait_time / 200.0 +       # Wait time (normalized to ~200 sec total)
            -0.2 * total_queue / 100.0 +            # Queue length
            +0.2 * throughput / 50.0 +              # Throughput
            -0.2 * wb_queue / 30.0 +                # Westbound priority
            phase_change_penalty * 0.1              # Smooth transitions
        )
        
        return reward
    
    def _get_total_wait_time(self):
        wait_time = 0
        for veh_id in traci.vehicle.getIDList():
            wait_time += traci.vehicle.getWaitingTime(veh_id)
        return wait_time
    
    def _get_total_queue_length(self):
        queue = 0
        for lane in self.lanes:
            try:
                queue += traci.lane.getLastStepHaltingNumber(lane)
            except:
                pass
        return queue
    
    def _get_throughput(self):
        # Vehicles that arrived at their destination
        return traci.simulation.getArrivedNumber()
    
    def _action_to_phase(self, action):
        """
        Map discrete action to SUMO phase index.
        
        Network phases:
            0: NS Green (37s)
            1: NS Yellow (3s)
            2: All Red (1s)
            3: EW Green (35s)
            4: EW Yellow (3s)
            5: All Red (1s)
        
        Actions control main green phases only:
            0: NS Green (Phase 0)
            1: Request EW (triggers yellow -> all-red -> EW green)
            2: EW Green (Phase 3)
            3: Request NS (triggers yellow -> all-red -> NS green)
        """
        action_phase_map = {
            0: 0,   # NS Through Green
            1: 3,   # EW Through Green (will trigger transition)
            2: 3,   # EW Through Green
            3: 0,   # NS Through Green (will trigger transition)
        }
        return action_phase_map.get(action, 0)

    def close(self):
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False


# Alias for backward compatibility
MultiSensorOxfordEnv = OxfordHydeParkEnv
