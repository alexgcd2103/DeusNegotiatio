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
        Advanced reward function optimized for high-volume 2046 PM Peak.
        Logic: R = -Pressure - sum(WaitTime^2) - StagnationPenalty + Throughput
        """
        # 1. Differential Pressure (Incoming - Outgoing)
        pressure = 0
        incoming_lanes = [l for l in self.lanes if '_in_' in l]
        outgoing_lanes = ['NB_out', 'SB_out', 'EB_out', 'WB_out']
        
        for lane in incoming_lanes:
            pressure += traci.lane.getLastStepHaltingNumber(lane)
        for lane in outgoing_lanes:
            # Outgoing pressure is negative (we want cars out)
            try:
                pressure -= traci.lane.getLastStepHaltingNumber(lane + "_0") # index 0 usually
            except:
                pass

        # 2. Squared Waiting Time Penalty (Penalize long-tail delays)
        total_squared_wait = 0
        current_vehicles = traci.vehicle.getIDList()
        stagnation_penalty = 0
        
        for veh_id in current_vehicles:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            total_squared_wait += (wait_time / 100.0) ** 2 # Scale to keep values manageable
            
            # 3. Stagnation Tracking
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:
                self.veh_stagnation[veh_id] = self.veh_stagnation.get(veh_id, 0) + 1
                if self.veh_stagnation[veh_id] > 20: # Over 20 simulation steps (~2 minutes)
                    stagnation_penalty += 5.0
            else:
                self.veh_stagnation[veh_id] = 0
        
        # Cleanup stagnation dict for departed vehicles
        departed = traci.simulation.getArrivedIDList()
        for veh_id in departed:
            if veh_id in self.veh_stagnation:
                del self.veh_stagnation[veh_id]

        throughput = self._get_throughput()
        
        # Phase change penalty (slightly increased to prevent flickering)
        phase_change_penalty = -1.0 if action != self.last_action else 0.0
        
        # Final Reward Composition
        reward = (
            -1.0 * pressure / 50.0 +           # Pressure (normalized)
            -2.0 * total_squared_wait / 10.0 +  # Squared wait penalty
            -1.0 * stagnation_penalty +        # Stagnation penalty
            +10.0 * throughput +               # High incentive for clearing cars
            phase_change_penalty * 0.2          # Smooth transitions
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
            0: 0,   # NS Through Green (Phase 0 in net.xml)
            1: 4,   # EW Through Green (Phase 4 in net.xml)
            2: 4,   # EW Through Green (Phase 4 in net.xml)
            3: 0,   # NS Through Green (Phase 0 in net.xml)
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
