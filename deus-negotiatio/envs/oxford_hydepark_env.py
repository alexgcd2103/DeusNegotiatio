import sumo_rl
import sumo_rl
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import time

import os
import os
from sensors.lidar_simulator import LidarSimulator
from sensors.infrared_simulator import InfraredSimulator
from sensors.motion_detector import MotionDetector
from sensors.environmental_effects import EnvironmentalEffects
class OxfordHydeParkEnv(gym.Env):
    """
    Custom Gym environment for Oxford-Hyde Park intersection
    """
    
    def __init__(self, 
                 net_file='oxford_hydepark.net.xml',
                 route_file='routes_2046_am_peak_calibrated.rou.xml',
                 use_gui=False,
                 num_seconds=3600,
                 delta_time=5,
                 yellow_time=3,
                 min_green=10,
                 max_green=50):
        
        super().__init__()
        
        # Resolve paths relative to this file's parent directory (assuming files are in deus-negotiatio root)
        # envs/oxford_hydepark_env.py -> parent is envs/ -> parent is deus-negotiatio/
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.net_file = os.path.join(base_path, net_file)
        self.route_file = os.path.join(base_path, route_file)
        
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        
        # Traffic light ID from network
        self.ts_id = 'oxford_hydepark_tl'

        # --- Sensor Simulators ---
        self.lidar = LidarSimulator()
        self.infrared = InfraredSimulator()
        self.motion = MotionDetector()
        self.weather = EnvironmentalEffects()
        
        # Define observation space (Multi-Sensor)
        # 1. SUMO Ground Truth (60 features)
        # 2. LiDAR Occupancy (8x8 = 64 features)
        # 3. Thermal Features (4 quadrants + 1 emergency = 5 features)
        # 4. Motion Features (4 vectors + 3 class dist = 7 features)
        # 5. Weather State (4 features)
        # 6. Time Features (2 features: sin/cos time)
        # Total: 60 + 64 + 5 + 7 + 4 + 2 = 142 features
        # Padding to 156 for future expansion/safety
        
        self.obs_dim = 156
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(self.obs_dim,),  
            dtype=np.float32
        )
        
        # 4 actions for 4 main phases
        self.action_space = spaces.Discrete(4)
        
        self.sumo_running = False
        self.simulation_step = 0
        
    def reset(self):
        """Reset simulation to initial state"""
        if self.sumo_running:
            traci.close()
            time.sleep(1) 
        
        # Resolve additional files
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        additional_files = [
            os.path.join(base_path, 'detectors.add.xml'),
            os.path.join(base_path, 'pedestrian_crossings.add.xml')
        ]

        # Use full path to SUMO binary from SUMO_HOME
        sumo_home = os.environ.get('SUMO_HOME', '')
        sumo_binary = 'sumo-gui' if self.use_gui else 'sumo'
        if sumo_home:
            sumo_binary = os.path.join(sumo_home, 'bin', sumo_binary)
        
        sumo_cmd = [
            sumo_binary,
            '-n', self.net_file,
            '-r', self.route_file,
            '--additional-files', ','.join(additional_files),
            '--no-warnings',
            '--no-step-log',
            '--time-to-teleport', '-1',
            '--waiting-time-memory', '1000',
            '--max-depart-delay', '0'
        ]
        
        traci.start(sumo_cmd)
        self.sumo_running = True
        self.simulation_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, terminated, truncated, info"""
        
        # Apply action (set traffic light phase)
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        target_phase = self._action_to_phase(action)
        
        if self._phase_to_action(current_phase) != action:
            traci.trafficlight.setPhase(self.ts_id, target_phase)
        
        # Simulate for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.simulation_step += 1
            # Update weather dynamics
            self.weather.step()
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        terminated = self.simulation_step >= self.num_seconds
        truncated = False
        
        info = {
            'total_wait_time': self._get_total_wait_time(),
            'queue_length': self._get_total_queue_length(),
            'throughput': self._get_throughput(),
            'pedestrian_wait': self._get_pedestrian_wait_time()
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Construct multi-sensor state vector"""
        # 1. SUMO Ground Truth (Base)
        state = []
        
        # Lane-level features for all 16 approach lanes
        lanes = [
            'oxford_eb_approach_0', 'oxford_eb_approach_1', 
            'oxford_eb_approach_2', 'oxford_eb_approach_3',
            'oxford_wb_approach_0', 'oxford_wb_approach_1',
            'oxford_wb_approach_2', 'oxford_wb_approach_3',
            'hydepark_nb_approach_0', 'hydepark_nb_approach_1',
            'hydepark_nb_approach_2', 'hydepark_nb_approach_3',
            'hydepark_sb_approach_0', 'hydepark_sb_approach_1',
            'hydepark_sb_approach_2', 'hydepark_sb_approach_3'
        ]
        
        # Collect raw vehicle data for sensors
        all_vehicles = []
        for veh_id in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(veh_id)
            # Center relative to intersection
            # We will calculate the centroid of all vehicles to center the view dynamically
            # This avoids needing a specific hardcoded junction ID
            
            all_vehicles.append({
                'id': veh_id,
                'x': x,
                'y': y,
                'speed': traci.vehicle.getSpeed(veh_id),
                'acceleration': traci.vehicle.getAcceleration(veh_id),
                'angle': np.radians(traci.vehicle.getAngle(veh_id)),
                'length': traci.vehicle.getLength(veh_id)
            })
            
        # Center coordinates relative to average vehicle position (dynamic centering)
        # This keeps the 'sensor' focused on the traffic cluster
        if all_vehicles:
            mean_x = np.mean([v['x'] for v in all_vehicles])
            mean_y = np.mean([v['y'] for v in all_vehicles])
            for v in all_vehicles:
                v['x'] -= mean_x
                v['y'] -= mean_y
        
        
        # -- Feature Construction --
        
        # 1. Base SUMO Features (60)
        for lane in lanes:
            try:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                avg_speed = traci.lane.getLastStepMeanSpeed(lane)
                occupancy = traci.lane.getLastStepOccupancy(lane)
            except:
                queue, avg_speed, occupancy = 0, 0, 0
            state.extend([queue, avg_speed, occupancy])
            
        current_phase = traci.trafficlight.getPhase(self.ts_id)
        phase_encoding = np.zeros(12) 
        if current_phase < 12:
            phase_encoding[current_phase] = 1.0
        state.extend(phase_encoding)
        
        # 2. LiDAR Features (64)
        lidar_data = self.lidar.scan(all_vehicles)
        lidar_data = self.weather.apply_to_lidar(lidar_data)
        state.extend(lidar_data['occupancy_grid'].flatten())
        
        # 3. Infrared Features (5)
        ir_data = self.infrared.capture(all_vehicles)
        state.extend(ir_data['quadrant_heat'])
        state.append(ir_data['emergency_heat'])
        
        # 4. Motion Features (7)
        motion_data = self.motion.detect(all_vehicles)
        motion_data = self.weather.apply_to_motion(motion_data)
        state.extend(motion_data['motion_vectors']) # 4
        state.extend(motion_data['class_dist'])     # 3
        
        # 5. Weather Features (4)
        state.extend(self.weather.get_state_feature())
        
        # 6. Time Features (2)
        sim_time = traci.simulation.getTime()
        day_progress = (sim_time % 86400) / 86400.0 * 2 * np.pi
        state.append(np.sin(day_progress))
        state.append(np.cos(day_progress))
        
        # Pad to match obs_dim
        current_len = len(state)
        if current_len < self.obs_dim:
            state.extend([0.0] * (self.obs_dim - current_len))
        elif current_len > self.obs_dim:
            state = state[:self.obs_dim]
            
        return np.array(state, dtype=np.float32)
    
    def _compute_reward(self):
        """Multi-objective reward function"""
        total_wait_time = self._get_total_wait_time()
        total_queue = self._get_total_queue_length()
        throughput = self._get_throughput()
        ped_wait = self._get_pedestrian_wait_time()
        
        reward = (
            -0.4 * total_wait_time / 100.0 +
            -0.2 * total_queue / 50.0 +
            0.3 * throughput / 20.0 +
            -0.1 * ped_wait / 30.0
        )
        return reward
    
    def _get_total_wait_time(self):
        wait_time = 0
        for veh_id in traci.vehicle.getIDList():
            wait_time += traci.vehicle.getWaitingTime(veh_id)
        return wait_time
    
    def _get_total_queue_length(self):
        queue = 0
        for lane in traci.lane.getIDList():
            if 'approach' in lane:
                queue += traci.lane.getLastStepHaltingNumber(lane)
        return queue
    
    def _get_throughput(self):
        return traci.simulation.getDepartedNumber()
    
    def _get_pedestrian_wait_time(self):
        ped_ids = traci.person.getIDList()
        if len(ped_ids) == 0:
            return 0
        total_wait = sum(traci.person.getWaitingTime(pid) for pid in ped_ids)
        return total_wait / len(ped_ids)
    
    def _action_to_phase(self, action):
        """Map discrete action to SUMO phase index"""
        action_phase_map = {
            0: 2,   # Oxford EW Through (Phase 2)
            1: 6,   # Oxford EW Left (Phase 6)
            2: 0,   # Hyde Park NS Through (Phase 0)
            3: 4,   # Hyde Park NS Left (Phase 4)
        }
        return action_phase_map.get(action, 0)
    
    def _phase_to_action(self, phase):
        """Reverse mapping"""
        phase_action_map = {
            0: 2, 1: 2, 
            2: 0, 3: 0, 
            4: 3, 5: 3, 
            6: 1, 7: 1
        }
        return phase_action_map.get(phase, 0)

    def close(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False
