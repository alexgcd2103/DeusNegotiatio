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
                 max_green=55,
                 extra_sumo_args=None):
        
        super().__init__()
        self.extra_sumo_args = extra_sumo_args or []
        
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
            # Eastbound (4 lanes)
            'EB_in_0', 'EB_in_1', 'EB_in_2', 'EB_in_3',
            # Westbound (4 lanes)
            'WB_in_0', 'WB_in_1', 'WB_in_2', 'WB_in_3'
        ]
        self.num_lanes = 16

        # --- Sensor Simulators ---
        self.lidar = LidarSimulator()
        self.infrared = InfraredSimulator()
        self.motion = MotionDetector()
        self.weather = EnvironmentalEffects()
        
        # Observation space (172-dim multi-sensor)
        # Lane features: 16 lanes × 3 features = 48
        # Phase encoding: 6 phases = 6
        # LiDAR grid: 8×8 = 64
        # Infrared: 5
        # Motion: 7
        # Weather: 4
        # Time: 2
        # TiQ Features: 16 lanes = 16
        # Total: 152 + 20? Wait.
        # 48 (SUMO) + 64 (LiDAR) + 5 (IR) + 7 (Motion) + 4 (Weather) + 2 (Time) + 16 (TiQ) = 146? 
        # Let's re-calculate:
        # SUMO Lanes: 16 * 3 = 48
        # Phase: 6
        # LiDAR: 64
        # IR: 5
        # Motion: 7
        # Weather: 4
        # Time: 2
        # TiQ (avg time in queue per lane): 16
        # Total = 48+6+64+5+7+4+2+16 = 151.
        # I will set obs_dim to 172 as planned to allow for future expansion and padding. (151 + 21 padding)
        self.obs_dim = 172
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
        self.last_phase = 0
        self.action_counts = {0:0, 1:0, 2:0, 3:0}
        
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
        ] + self.extra_sumo_args
        
        traci.start(sumo_cmd)
        self.sumo_running = True
        self.simulation_step = 0
        self.last_action = 0
        
        # Reset weather and vehicle tracking
        self.weather = EnvironmentalEffects()
        self.veh_stagnation = {} # veh_id -> steps at speed < 0.1
        self.veh_tiq = {}        # veh_id -> accumulated wait time in current queue
        self.action_counts = {0:0, 1:0, 2:0, 3:0}
        self.last_phase = traci.trafficlight.getPhase(self.ts_id)
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, terminated, truncated, info"""
        
        # Apply action (set traffic light phase)
        target_phase = self._action_to_phase(action)
        transition_steps = self._set_phase(target_phase)
        self.action_counts[action] += 1
        
        # Simulate for remaining delta_time seconds to ensure exact 5s decision intervals
        remaining_steps = max(0, self.delta_time - transition_steps)
        for _ in range(remaining_steps):
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
            'vehicles_in_sim': len(traci.vehicle.getIDList()),
            'action_distribution': {a: c / max(1, sum(self.action_counts.values())) for a, c in self.action_counts.items()}
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
        progress = (sim_time % 3600) / 3600.0 * 2 * np.pi
        state.append(np.sin(progress))
        state.append(np.cos(progress))
        
        # --- 8. Time-in-Queue (TiQ) Features (16) ---
        # Calculate average TiQ per lane
        lane_tiqs = {lane: [] for lane in self.lanes}
        for veh_id in traci.vehicle.getIDList():
            try:
                lane = traci.vehicle.getLaneID(veh_id)
                if lane in lane_tiqs:
                    wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    lane_tiqs[lane].append(wait_time)
            except:
                pass
        
        for lane in self.lanes:
            avg_tiq = np.mean(lane_tiqs[lane]) if lane_tiqs[lane] else 0.0
            state.append(avg_tiq / 300.0) # Normalize to 5 minutes
        
        # Pad to obs_dim
        current_len = len(state)
        if current_len < self.obs_dim:
            state.extend([0.0] * (self.obs_dim - current_len))
        elif current_len > self.obs_dim:
            state = state[:self.obs_dim]
            
        return np.array(state, dtype=np.float32)
    
    def _compute_reward(self, action):
        """
        Stabilized reward function for high-volume 2046 PM Peak.
        Prevents exploding rewards while keeping penalties for long-tail wait times.
        """
        # 1. Differential Pressure (Incoming - Outgoing)
        pressure = 0
        incoming_lanes = [l for l in self.lanes if '_in_' in l]
        # Map of outgoing edges to their primary entry lane
        outgoing_map = {'NB_out': 'NB_out_0', 'SB_out': 'SB_out_0', 'EB_out': 'EB_out_0', 'WB_out': 'WB_out_0'}
        
        for lane in incoming_lanes:
            pressure += traci.lane.getLastStepHaltingNumber(lane)
        for edge_id, lane_id in outgoing_map.items():
            try:
                pressure -= traci.lane.getLastStepHaltingNumber(lane_id)
            except:
                pass

        # 2. Log-Scaled Waiting Time Penalty
        # log2(1 + wait_time) provides a stable penalty that still discourages long waits
        total_wait_penalty = 0
        current_vehicles = traci.vehicle.getIDList()
        stagnation_count = 0
        
        for veh_id in current_vehicles:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            total_wait_penalty += np.log2(1 + wait_time)
            
            # 3. Stagnation Tracking (Improved criteria)
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:
                self.veh_stagnation[veh_id] = self.veh_stagnation.get(veh_id, 0) + 1
                if self.veh_stagnation[veh_id] > 20: # Over 20 simulation steps (~1 minute)
                    stagnation_count += 1
            else:
                self.veh_stagnation[veh_id] = 0
        
        # Cleanup stagnation dict for departed vehicles
        departed = traci.simulation.getArrivedIDList()
        for veh_id in departed:
            if veh_id in self.veh_stagnation:
                del self.veh_stagnation[veh_id]

        throughput = self._get_throughput()
        
        # Phase change penalty (conservative)
        phase_change_penalty = -1.0 if action != self.last_action else 0.0
        
        # Combined Reward (Normalized to roughly -10.0 to +10.0 per step)
        # Pressure is now exponential to create massive gradients for high-congestion states
        # exp(p/10) - 1 gives 0 at p=0, ~1.7 at p=10, ~6.4 at p=20, ~21 at p=30.
        exponential_penalty = np.exp(pressure / 10.0) - 1.0
        
        reward = (
            -1.0 * exponential_penalty +        # High-drama exponential pressure
            -1.0 * total_wait_penalty / 100.0 + # Wait penalty
            -2.0 * stagnation_count / 10.0 +    # Stagnation penalty
            +5.0 * throughput +                 # Incentive for clearing cars
            phase_change_penalty * 0.1
        )
        # Final Global Scaling to pull reward into stable range (~Hundreds per episode)
        # Increased to 0.05 from 0.01 to provide sharper gradient for pattern synthesis
        reward_scaled = reward * 0.05
        
        return float(reward_scaled)
    
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
        
        Action Mapping (Oxford-Hyde Park):
            0: NS Green Through (Phase 0)
            1: NS Left Turn (Phase 2)
            2: EW Green Through (Phase 4)
            3: All-Red/Wait Default (Keep current or safer phase)
        """
        action_phase_map = {
            0: 0,   # NS Through
            1: 2,   # NS Left
            2: 4,   # EW Through
            3: 0,   # Stay NS Through (Safest default)
        }
        return action_phase_map.get(action, 0)

    def _set_phase(self, target_phase):
        """
        Safety-first phase setter.
        Triggers yellow/all-red transitions if changing between different Green phases.
        Returns the number of simulation steps spent in transitions.
        """
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        
        if current_phase == target_phase:
            traci.trafficlight.setPhase(self.ts_id, target_phase)
            return 0

        # Determine intermediate yellow phase
        # Network logic: 0(G)->1(y)->2(G)->3(y)->4(G)->5(y)->0
        yellow_phase = current_phase + 1
        if yellow_phase > 5: yellow_phase = 0
        
        traci.trafficlight.setPhase(self.ts_id, yellow_phase)
        
        steps_spent = 0
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self.simulation_step += 1
            self.weather.step()
            steps_spent += 1
        
        # Set target phase
        traci.trafficlight.setPhase(self.ts_id, target_phase)
        self.last_phase = target_phase
        return steps_spent

    def close(self):
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False


# Alias for backward compatibility
MultiSensorOxfordEnv = OxfordHydeParkEnv
