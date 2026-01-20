# Multi-Sensor Traffic Light Agent Training Implementation Guide

## Executive Overview

Transform your traffic light agent from a single-parameter system into a **singularity intersection controller** capable of synthesizing patterns across multiple sensor modalities and environmental conditions. This guide covers simulated LiDAR, infrared, motion sensors, and environmental factors (glare, snow/ice, precipitation) integrated with your SUMO-based training pipeline.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│           CARLA-SUMO Co-Simulation Framework                │
│  (Real-time 3D rendering + Traffic flow simulation)         │
└────────────────┬────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼─────────────┐    ┌─────▼──────────────┐
│  SUMO Traffic   │    │  CARLA 3D          │
│  Flow Control   │    │  Environment       │
│  (agents + TLC) │    │  (sensors + fx)    │
└───┬─────────────┘    └─────┬──────────────┘
    │                         │
    └────────────┬────────────┘
                 │
     ┌───────────▼───────────┐
     │  Sensor Simulation    │
     │  Layer               │
     │ ┌─────────────────┐  │
     │ │ • LiDAR Sim     │  │
     │ │ • Infrared Sim  │  │
     │ │ • Motion Detect │  │
     │ │ • Vehicle Clss  │  │
     │ └─────────────────┘  │
     └───────────┬───────────┘
                 │
     ┌───────────▼────────────────┐
     │  Environmental Effects     │
     │  Processing Pipeline       │
     │ ┌──────────────────────┐   │
     │ │ • Glare Simulation   │   │
     │ │ • Weather (snow/ice) │   │
     │ │ • Precipitation      │   │
     │ │ • Lighting Changes   │   │
     │ │ • Road Conditions    │   │
     │ └──────────────────────┘   │
     └───────────┬────────────────┘
                 │
     ┌───────────▼───────────────┐
     │  Synthetic Dataset       │
     │  Generation             │
     │ ┌─────────────────────┐  │
     │ │ Multi-channel data  │  │
     │ │ with annotations    │  │
     │ │ and metadata        │  │
     │ └─────────────────────┘  │
     └───────────┬───────────────┘
                 │
     ┌───────────▼──────────────────┐
     │  RL Agent Training            │
     │  (Multi-Agent Q-Learning)     │
     │ ┌──────────────────────────┐  │
     │ │ Reward: Wait time        │  │
     │ │         throughput       │  │
     │ │         environmental fx │  │
     │ └──────────────────────────┘  │
     └───────────┬──────────────────┘
                 │
        ┌────────▼────────┐
        │  Agent Model    │
        │  (Singularity   │
        │   Controller)   │
        └─────────────────┘
```

---

## Part 1: CARLA-SUMO Co-Simulation Setup

### 1.1 Installation Prerequisites

```bash
# Ubuntu 20.04+ or similar
# SUMO 1.15+
# CARLA 0.9.14+
# Python 3.8+

# Install SUMO
sudo apt-get install sumo sumo-tools sumo-doc

# Install CARLA (precompiled)
# Download from: https://github.com/carla-simulator/carla/releases
# Or use pip:
pip install carla==0.9.14

# Install co-simulation dependencies
pip install sumo-rl stable-baselines3 gymnasium torch torchvision
pip install numpy pandas scikit-learn opencv-python
pip install scipy matplotlib seaborn
```

### 1.2 Bridge Configuration

Create `carla_sumo_bridge.py` to synchronize simulations:

```python
import carla
import subprocess
import socket
import time
from typing import Dict, Tuple
import xml.etree.ElementTree as ET

class CARLASUMOBridge:
    """
    Bidirectional bridge between CARLA and SUMO for co-simulation.
    Handles vehicle position sync, sensor data routing, and signal control.
    """
    
    def __init__(self, carla_host='localhost', carla_port=2000, 
                 sumo_port=8813, sumo_net_file='network.net.xml'):
        self.carla_client = carla.Client(carla_host, carla_port)
        self.carla_world = self.carla_client.get_world()
        self.carla_map = self.carla_world.get_map()
        
        # SUMO configuration
        self.sumo_port = sumo_port
        self.sumo_net_file = sumo_net_file
        self.sumo_started = False
        self.tick_count = 0
        self.sync_period = 0.05  # 50ms sync
        
        # Vehicle tracking
        self.carla_to_sumo_vehicles = {}  # carla_id -> sumo_id
        self.sumo_to_carla_vehicles = {}  # sumo_id -> carla_id
        self.traffic_light_map = {}  # sumo_tl_id -> carla_tl_actor
        
    def start_sumo(self, sumo_config_file: str):
        """Launch SUMO with TraCI interface enabled."""
        cmd = [
            'sumo',
            '-c', sumo_config_file,
            '--remote-port', str(self.sumo_port),
            '--step-length', '0.05',
            '--default.speeddev', '0.0',  # Deterministic
            '--seed', '42'  # Reproducible
        ]
        subprocess.Popen(cmd)
        self.sumo_started = True
        time.sleep(3)  # Wait for SUMO to start
        
    def synchronize_tick(self) -> Dict:
        """Single synchronization step between CARLA and SUMO."""
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.sync_period
        self.carla_world.apply_settings(settings)
        
        snapshot = self.carla_world.tick()
        self.tick_count += 1
        
        return {
            'timestamp': snapshot.timestamp.elapsed_seconds,
            'tick': self.tick_count,
            'frame': snapshot.frame
        }
```

---

## Part 2: Sensor Simulation Layer

### 2.1 LiDAR Simulation

LiDAR generates 3D point clouds showing distance and intensity. Simulate via CARLA's native LiDAR + custom noise injection.

```python
import numpy as np
from typing import List, Tuple
import open3d as o3d

class LiDARSimulator:
    """
    Simulates automotive LiDAR sensor with configurable parameters.
    Outputs: point clouds, intensity maps, range images
    """
    
    def __init__(self, 
                 max_range: float = 100.0,
                 horizontal_fov: float = 120.0,
                 vertical_fov: float = 25.0,
                 num_channels: int = 64,
                 points_per_second: int = 1_200_000,
                 noise_stddev: float = 0.02):
        """
        Args:
            max_range: Maximum sensing distance in meters
            horizontal_fov: Horizontal field of view (degrees)
            vertical_fov: Vertical field of view (degrees)
            num_channels: Number of laser channels (mimics Velodyne 64-channel)
            points_per_second: Point cloud generation rate
            noise_stddev: Gaussian noise standard deviation (meters)
        """
        self.max_range = max_range
        self.horizontal_fov = np.radians(horizontal_fov)
        self.vertical_fov = np.radians(vertical_fov)
        self.num_channels = num_channels
        self.points_per_second = points_per_second
        self.noise_stddev = noise_stddev
        
    def capture(self, carla_lidar_sensor) -> Dict:
        """Process CARLA LiDAR data with noise and filtering."""
        # Get raw point cloud from CARLA sensor
        point_cloud = carla_lidar_sensor.get_point_cloud()
        
        if len(point_cloud) == 0:
            return {'points': np.array([]), 'intensity': np.array([])}
        
        # Extract xyz and intensity
        xyz = np.array([list(p) for p in point_cloud])[:, :3]
        intensity = np.array([list(p) for p in point_cloud])[:, 3]
        
        # Add Gaussian noise (realistic sensor error)
        xyz_noisy = xyz + np.random.normal(0, self.noise_stddev, xyz.shape)
        
        # Range filtering (points beyond max_range are removed)
        ranges = np.linalg.norm(xyz_noisy, axis=1)
        valid_mask = ranges < self.max_range
        xyz_filtered = xyz_noisy[valid_mask]
        intensity_filtered = intensity[valid_mask]
        
        # Return structured data
        return {
            'points': xyz_filtered,
            'intensity': intensity_filtered,
            'ranges': ranges[valid_mask],
            'num_points': len(xyz_filtered),
            'timestamp': time.time()
        }
    
    def extract_objects(self, lidar_data: Dict) -> List[Dict]:
        """
        Clustering to identify vehicles/pedestrians from point cloud.
        Uses DBSCAN for spatial clustering.
        """
        from sklearn.cluster import DBSCAN
        
        points = lidar_data['points']
        
        if len(points) < 5:
            return []
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(points)
        labels = clustering.labels_
        
        objects = []
        for label in set(labels):
            if label == -1:  # Skip noise
                continue
            
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            extent = np.max(np.abs(cluster_points - centroid), axis=0)
            
            objects.append({
                'centroid': centroid,
                'extent': extent,  # [length, width, height]
                'num_points': len(cluster_points),
                'type': 'unknown'  # Will be classified
            })
        
        return objects
```

### 2.2 Infrared (Thermal) Camera Simulation

Thermal imaging sensitive to heat signatures and temperature gradients.

```python
class InfraredSimulator:
    """
    Simulates thermal/infrared imaging for traffic monitoring.
    Outputs: temperature maps, heat signatures for vehicles
    """
    
    def __init__(self,
                 resolution: Tuple[int, int] = (640, 480),
                 fov: float = 90.0,
                 ambient_temp: float = 20.0,  # Celsius
                 min_temp: float = -40.0,
                 max_temp: float = 150.0):
        """
        Args:
            resolution: Output resolution (width, height)
            fov: Field of view (degrees)
            ambient_temp: Background ambient temperature
            min_temp/max_temp: Thermal range
        """
        self.resolution = resolution
        self.fov = fov
        self.ambient_temp = ambient_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        
    def capture_thermal(self, vehicles: List, environment_data: Dict) -> np.ndarray:
        """
        Generate thermal image from vehicle positions and engine states.
        
        Args:
            vehicles: List of vehicle objects with position/velocity
            environment_data: Contains ambient_temp, engine_activity, etc.
        
        Returns:
            Thermal image (640x480) with temperature values
        """
        thermal_image = np.ones(self.resolution) * self.ambient_temp
        
        for vehicle in vehicles:
            # Vehicle engine temperature correlates with movement
            speed = np.linalg.norm(vehicle.velocity)
            engine_temp = self.ambient_temp + (speed * 2)  # Hotter when moving
            
            # Braking increases heat
            if hasattr(vehicle, 'brake_force') and vehicle.brake_force > 0:
                engine_temp += 15
            
            # Project vehicle center to image coordinates
            screen_pos = self._world_to_screen(vehicle.position)
            
            if self._in_frame(screen_pos):
                # Draw thermal blob (Gaussian)
                y, x = int(screen_pos[1]), int(screen_pos[0])
                sigma = 20
                
                yy, xx = np.ogrid[:self.resolution[1], :self.resolution[0]]
                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                thermal_image += gaussian * (engine_temp - self.ambient_temp)
        
        # Clamp to thermal range
        thermal_image = np.clip(thermal_image, self.min_temp, self.max_temp)
        
        return thermal_image
    
    def _world_to_screen(self, world_pos) -> Tuple[float, float]:
        """Convert 3D world position to 2D screen coordinates."""
        # Simplified projection; would use actual camera matrix in practice
        fov_rad = np.radians(self.fov)
        focal_length = self.resolution[0] / (2 * np.tan(fov_rad / 2))
        
        x_screen = (world_pos[0] / max(world_pos[2], 0.1)) * focal_length + self.resolution[0] / 2
        y_screen = (world_pos[1] / max(world_pos[2], 0.1)) * focal_length + self.resolution[1] / 2
        
        return (x_screen, y_screen)
    
    def _in_frame(self, pos: Tuple) -> bool:
        """Check if position is within frame bounds."""
        return 0 <= pos[0] < self.resolution[0] and 0 <= pos[1] < self.resolution[1]
```

### 2.3 Motion Detection & Vehicle Classification

Optical flow + background subtraction for motion detection, ML model for classification.

```python
class MotionSensorSimulator:
    """
    Motion detection via optical flow and vehicle classification.
    Simulates smart camera-based detection at intersections.
    """
    
    def __init__(self, 
                 model_type: str = 'yolov5s',
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45):
        """
        Args:
            model_type: YOLOv5 variant ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Detection confidence
            nms_threshold: Non-max suppression threshold
        """
        try:
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
            self.model.conf = confidence_threshold
            self.model.iou = nms_threshold
        except:
            print("Warning: YOLOv5 not available. Using dummy detector.")
            self.model = None
        
        self.prev_frame = None
        self.prev_detections = []
        
    def detect_vehicles(self, rgb_image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in RGB image using YOLOv5.
        
        Returns:
            List of detections: {bbox, confidence, class, centroid, velocity}
        """
        if self.model is None:
            return self._dummy_detect(rgb_image)
        
        results = self.model(rgb_image)
        detections = []
        
        for *bbox, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = [int(v.item()) for v in bbox]
            confidence = conf.item()
            class_id = int(cls.item())
            
            # Vehicle class IDs in COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
            if class_id in [2, 3, 5, 7]:
                centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                
                # Estimate velocity from frame-to-frame movement
                velocity = self._estimate_velocity(centroid)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class': class_id,
                    'centroid': centroid,
                    'velocity': velocity,
                    'area': (x2 - x1) * (y2 - y1),
                    'heading': self._estimate_heading(velocity)
                })
        
        self.prev_detections = detections
        self.prev_frame = rgb_image.copy()
        
        return detections
    
    def _estimate_velocity(self, centroid: np.ndarray) -> np.ndarray:
        """Optical flow: compare centroid positions frame-to-frame."""
        if self.prev_detections and len(self.prev_detections) > 0:
            # Simple nearest neighbor matching
            prev_centroids = np.array([d['centroid'] for d in self.prev_detections])
            distances = np.linalg.norm(prev_centroids - centroid, axis=1)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < 50:  # Reasonable tracking distance
                velocity = centroid - self.prev_detections[nearest_idx]['centroid']
                return velocity
        
        return np.array([0.0, 0.0])
    
    def _estimate_heading(self, velocity: np.ndarray) -> float:
        """Calculate heading angle from velocity vector."""
        return np.arctan2(velocity[1], velocity[0])
    
    def _dummy_detect(self, rgb_image: np.ndarray) -> List[Dict]:
        """Fallback dummy detector for testing."""
        return []
```

---

## Part 3: Environmental Effects Pipeline

### 3.1 Weather & Lighting Effects

```python
class EnvironmentalEffectsSimulator:
    """
    Simulates weather, lighting, and road conditions.
    Modifies sensor data to reflect real-world environmental challenges.
    """
    
    def __init__(self):
        self.weather_state = {
            'precipitation': 0.0,      # 0-100%
            'wind_speed': 0.0,         # km/h
            'visibility': 1000.0,      # meters
            'ambient_temp': 20.0,      # Celsius
            'road_wetness': 0.0,       # 0-1
        }
        self.time_of_day = 12.0  # 0-24 hours
        self.sun_position = None
        
    def update_weather(self, weather_type: str, intensity: float = 0.5):
        """
        Update weather condition.
        
        Args:
            weather_type: 'clear', 'rain', 'snow', 'fog', 'hail'
            intensity: 0-1 scale
        """
        if weather_type == 'rain':
            self.weather_state['precipitation'] = intensity * 100
            self.weather_state['road_wetness'] = min(1.0, intensity * 1.5)
            self.weather_state['visibility'] = 500 - (intensity * 400)
            
        elif weather_type == 'snow':
            self.weather_state['precipitation'] = intensity * 80
            self.weather_state['ambient_temp'] = min(0, -5 * intensity)
            self.weather_state['visibility'] = 300 - (intensity * 250)
            self.weather_state['road_wetness'] = 0.7 + (intensity * 0.3)
            
        elif weather_type == 'fog':
            self.weather_state['visibility'] = 100 + (intensity * 200)
            
        elif weather_type == 'clear':
            self.weather_state['precipitation'] = 0
            self.weather_state['visibility'] = 1000
            self.weather_state['road_wetness'] = 0
    
    def apply_sun_glare(self, 
                       rgb_image: np.ndarray,
                       sun_altitude: float,
                       sun_azimuth: float,
                       camera_direction: np.ndarray) -> np.ndarray:
        """
        Simulate sun glare when camera is oriented toward sun.
        High-intensity bright spots reduce visibility.
        
        Args:
            rgb_image: Input RGB image
            sun_altitude: Sun angle above horizon (degrees)
            sun_azimuth: Sun direction (0-360)
            camera_direction: Camera viewing direction
        
        Returns:
            Image with glare artifacts
        """
        # Calculate glare intensity based on sun-camera angle
        sun_pos = np.array([
            np.cos(np.radians(sun_altitude)) * np.sin(np.radians(sun_azimuth)),
            np.sin(np.radians(sun_altitude)),
            np.cos(np.radians(sun_altitude)) * np.cos(np.radians(sun_azimuth))
        ])
        
        dot_product = np.dot(sun_pos, camera_direction / np.linalg.norm(camera_direction))
        glare_intensity = max(0, dot_product) ** 0.5  # Smooth falloff
        
        if glare_intensity > 0.3:
            # Add lens flare (bright center, fading rings)
            h, w = rgb_image.shape[:2]
            y_center, x_center = h // 2, w // 2
            
            yy, xx = np.ogrid[:h, :w]
            distance = np.sqrt((xx - x_center)**2 + (yy - y_center)**2)
            
            # Create glare bloom
            glare_map = np.exp(-distance**2 / (2 * (100 ** 2))) * glare_intensity * 150
            
            # Apply to all channels
            for c in range(3):
                rgb_image[:, :, c] = np.clip(
                    rgb_image[:, :, c] + glare_map, 0, 255
                ).astype(np.uint8)
        
        return rgb_image
    
    def apply_rain_effect(self, 
                         rgb_image: np.ndarray,
                         lidar_data: Dict,
                         intensity: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Apply rain effects to camera image and reduce LiDAR range.
        
        Effects:
        - Raindrops on lens (random noise)
        - Reduced contrast
        - Color shift (blueish tint)
        - LiDAR attenuation (particles scatter laser)
        """
        rain_image = rgb_image.copy().astype(float)
        
        # Rain droplets (random pixels)
        h, w = rain_image.shape[:2]
        num_drops = int(h * w * intensity * 0.001)
        
        drop_y = np.random.randint(0, h, num_drops)
        drop_x = np.random.randint(0, w, num_drops)
        rain_image[drop_y, drop_x] = [200, 200, 200]  # Light gray
        
        # Reduced contrast
        rain_image = rain_image * (1 - intensity * 0.2) + 50 * intensity * 0.2
        
        # Color shift (more blue)
        rain_image[:, :, 0] *= (1 - intensity * 0.1)  # Less red
        rain_image[:, :, 1] *= (1 - intensity * 0.05) # Slightly less green
        rain_image[:, :, 2] *= (1 + intensity * 0.15) # More blue
        
        rain_image = np.clip(rain_image, 0, 255).astype(np.uint8)
        
        # LiDAR attenuation: reduce range and add noise
        lidar_data_modified = lidar_data.copy()
        lidar_data_modified['ranges'] *= (1 - intensity * 0.3)  # 30% range reduction at heavy rain
        
        # Add particle scatter noise
        particle_noise = np.random.normal(0, intensity * 0.5, lidar_data['points'].shape)
        lidar_data_modified['points'] = lidar_data['points'] + particle_noise
        
        return rain_image, lidar_data_modified
    
    def apply_snow_ice(self,
                       rgb_image: np.ndarray,
                       lidar_data: Dict,
                       intensity: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Snow/ice reduces visibility and makes surfaces reflective.
        """
        snow_image = rgb_image.copy().astype(float)
        
        # Add snow particles (white noise)
        h, w = snow_image.shape[:2]
        num_particles = int(h * w * intensity * 0.002)
        
        snow_y = np.random.randint(0, h, num_particles)
        snow_x = np.random.randint(0, w, num_particles)
        snow_image[snow_y, snow_x] = [255, 255, 255]
        
        # Brighten image (reflective snow)
        snow_image = snow_image * (1 + intensity * 0.3)
        
        # Desaturate (snow is white)
        gray = np.mean(snow_image, axis=2, keepdims=True)
        snow_image = gray * (1 - intensity * 0.4) + snow_image * (intensity * 0.4)
        
        snow_image = np.clip(snow_image, 0, 255).astype(np.uint8)
        
        # LiDAR heavily affected by snow
        lidar_data_modified = lidar_data.copy()
        lidar_data_modified['ranges'] *= (1 - intensity * 0.5)  # 50% range reduction
        
        # Add thick scatter
        particle_noise = np.random.normal(0, intensity * 1.0, lidar_data['points'].shape)
        lidar_data_modified['points'] = lidar_data['points'] + particle_noise
        
        return snow_image, lidar_data_modified
```

---

## Part 4: Synthetic Dataset Generation

### 4.1 Multi-Modal Data Pipeline

```python
from dataclasses import dataclass
from pathlib import Path
import pickle

@dataclass
class SensorFrame:
    """Single synchronized frame from all sensors."""
    timestamp: float
    frame_id: int
    
    # Vision
    rgb_image: np.ndarray      # (480, 640, 3)
    thermal_image: np.ndarray  # (480, 640)
    
    # 3D
    lidar_points: np.ndarray   # (N, 3)
    lidar_intensity: np.ndarray # (N,)
    lidar_objects: List[Dict]  # Clustered objects
    
    # Detection
    vehicle_detections: List[Dict]  # Bounding boxes + classes
    motion_vectors: np.ndarray      # Optical flow
    
    # State
    traffic_state: Dict  # Vehicles, signal phases, queues
    weather_state: Dict  # Temperature, precipitation, visibility
    intersection_state: Dict  # Queue lengths, waiting vehicles
    
    # Metadata
    scenario_id: str
    weather_condition: str
    time_of_day: float
    sun_position: Tuple[float, float]

class DatasetGenerator:
    """
    Orchestrates multi-modal data collection from co-simulation.
    Outputs structured datasets for RL training.
    """
    
    def __init__(self, 
                 output_dir: Path = Path('./training_data'),
                 target_fps: int = 10,
                 max_frames_per_scenario: int = 1000):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_fps = target_fps
        self.max_frames_per_scenario = max_frames_per_scenario
        
        # Sensor instances
        self.lidar_sim = LiDARSimulator()
        self.ir_sim = InfraredSimulator()
        self.motion_sim = MotionSensorSimulator()
        self.env_sim = EnvironmentalEffectsSimulator()
        
        # Scenario tracking
        self.current_scenario_id = 0
        self.frame_buffer = []
        
    def collect_frame(self,
                     carla_world,
                     sumo_state,
                     sensors: Dict) -> SensorFrame:
        """
        Collect synchronized frame from all sources.
        
        Args:
            carla_world: CARLA world object
            sumo_state: SUMO traffic state (vehicles, signals)
            sensors: Dict of CARLA sensor objects (camera, lidar, thermal)
        
        Returns:
            SensorFrame with all modalities
        """
        timestamp = time.time()
        
        # 1. Vision: RGB + Thermal
        rgb_image = sensors['camera'].get_image()
        thermal_image = self.ir_sim.capture_thermal(
            sumo_state['vehicles'],
            self.env_sim.weather_state
        )
        
        # Apply weather effects
        if self.env_sim.weather_state['precipitation'] > 0:
            rgb_image, _ = self.env_sim.apply_rain_effect(
                rgb_image, {}, self.env_sim.weather_state['precipitation'] / 100
            )
        
        # Apply glare if sunny
        if self.env_sim.time_of_day > 8 and self.env_sim.time_of_day < 18:
            sun_alt = 45.0 + 30 * np.sin(np.radians((self.env_sim.time_of_day - 12) * 15))
            sun_az = (self.env_sim.time_of_day - 6) * 15
            camera_dir = np.array([0, 0, 1])  # Forward
            
            rgb_image = self.env_sim.apply_sun_glare(
                rgb_image, sun_alt, sun_az, camera_dir
            )
        
        # 2. 3D: LiDAR
        lidar_raw = self.lidar_sim.capture(sensors['lidar'])
        lidar_objects = self.lidar_sim.extract_objects(lidar_raw)
        
        # Apply weather attenuation
        if self.env_sim.weather_state['precipitation'] > 50:
            rgb_image, lidar_raw = self.env_sim.apply_snow_ice(
                rgb_image, lidar_raw, self.env_sim.weather_state['precipitation'] / 100
            )
        
        # 3. Detection & Motion
        detections = self.motion_sim.detect_vehicles(rgb_image)
        
        # Optical flow (simplified)
        motion_vectors = np.array([d['velocity'] for d in detections]) if detections else np.array([])
        
        # 4. Ground Truth State
        traffic_state = {
            'vehicles': sumo_state['vehicles'],
            'signals': sumo_state['traffic_lights'],
            'routes': sumo_state['vehicle_routes']
        }
        
        intersection_state = self._compute_intersection_state(sumo_state)
        
        # Create frame
        frame = SensorFrame(
            timestamp=timestamp,
            frame_id=len(self.frame_buffer),
            rgb_image=rgb_image,
            thermal_image=thermal_image,
            lidar_points=lidar_raw['points'],
            lidar_intensity=lidar_raw['intensity'],
            lidar_objects=lidar_objects,
            vehicle_detections=detections,
            motion_vectors=motion_vectors,
            traffic_state=traffic_state,
            weather_state=self.env_sim.weather_state.copy(),
            intersection_state=intersection_state,
            scenario_id=f"scenario_{self.current_scenario_id:04d}",
            weather_condition=self.env_sim.current_weather,
            time_of_day=self.env_sim.time_of_day,
            sun_position=(0, 0)
        )
        
        self.frame_buffer.append(frame)
        return frame
    
    def _compute_intersection_state(self, sumo_state: Dict) -> Dict:
        """
        Compute queue lengths and waiting times per approach.
        """
        state = {}
        for tl_id in sumo_state['traffic_lights']:
            incoming_lanes = sumo_state['lane_map'][tl_id]
            queue_lengths = {}
            wait_times = {}
            
            for lane_id in incoming_lanes:
                vehicles_in_lane = [v for v in sumo_state['vehicles'] 
                                   if v['lane'] == lane_id]
                queue_lengths[lane_id] = len(vehicles_in_lane)
                wait_times[lane_id] = np.mean([v['wait_time'] for v in vehicles_in_lane]) if vehicles_in_lane else 0
            
            state[tl_id] = {
                'queue_lengths': queue_lengths,
                'wait_times': wait_times,
                'total_vehicles': sum(queue_lengths.values())
            }
        
        return state
    
    def save_scenario(self, scenario_metadata: Dict):
        """
        Save collected frames and metadata to disk.
        """
        if not self.frame_buffer:
            return
        
        scenario_dir = self.output_dir / f"scenario_{self.current_scenario_id:04d}"
        scenario_dir.mkdir(exist_ok=True)
        
        # Save frames as NPZ (numpy compressed)
        frames_data = {
            'frames': self.frame_buffer,
            'metadata': scenario_metadata
        }
        
        np.savez_compressed(
            scenario_dir / 'frames.npz',
            **{f'frame_{i}': pickle.dumps(frame) for i, frame in enumerate(self.frame_buffer)}
        )
        
        # Save metadata JSON
        import json
        with open(scenario_dir / 'metadata.json', 'w') as f:
            json.dump({
                'scenario_id': self.current_scenario_id,
                'num_frames': len(self.frame_buffer),
                'duration_seconds': self.frame_buffer[-1].timestamp - self.frame_buffer[0].timestamp,
                'weather_condition': scenario_metadata.get('weather'),
                'time_of_day': scenario_metadata.get('time_of_day'),
                'high_traffic': scenario_metadata.get('high_traffic', False)
            }, f, indent=2)
        
        print(f"Saved scenario {self.current_scenario_id} with {len(self.frame_buffer)} frames")
        
        self.frame_buffer = []
        self.current_scenario_id += 1
```

---

## Part 5: RL Agent Training with Multi-Sensor Input

### 5.1 Observation Space Design

```python
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN

class MultiSensorTrafficEnv(gym.Env):
    """
    RL environment for traffic light control with multi-modal sensor inputs.
    
    Observation: Concatenated features from all sensors
    Action: Next traffic light phase
    Reward: Negative waiting time + throughput + environmental penalties
    """
    
    def __init__(self, 
                 sumo_net_file: str,
                 intersection_id: str,
                 max_episode_steps: int = 1000):
        super().__init__()
        
        self.intersection_id = intersection_id
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        
        # Action space: next traffic light phase (e.g., 4 phases)
        # 0=N-S green, 1=E-W green, 2=N-S yellow, 3=all red
        self.action_space = spaces.Discrete(4)
        
        # Observation space: multi-modal feature vector
        # Components:
        # - Queue lengths per lane: 4 lanes * 1 value = 4
        # - Vehicle detection confidence: 4 lanes * 1 value = 4
        # - LiDAR occupancy grid: 8x8 = 64
        # - Thermal hotspots: 4 values (one per quadrant)
        # - Motion vectors: 4 values (mean velocity per lane)
        # - Weather state: 5 values (temp, precipitation, visibility, wind, wetness)
        # - Time features: 3 values (hour, day_of_week, time_of_day)
        # - Pressure (wait time): 4 lanes
        # Total: 4+4+64+4+4+5+3+4 = 92 features
        
        self.obs_size = 92
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        
    def _extract_features(self, sensor_frame: SensorFrame) -> np.ndarray:
        """
        Convert SensorFrame into feature vector for RL agent.
        """
        features = []
        
        # 1. Queue lengths (normalized to 0-1)
        intersection_state = sensor_frame.intersection_state
        queue_features = []
        max_queue_length = 20  # Normalize
        
        for lane_id in sorted(intersection_state[self.intersection_id]['queue_lengths'].keys()):
            queue_len = intersection_state[self.intersection_id]['queue_lengths'][lane_id]
            queue_features.append(min(queue_len / max_queue_length, 1.0))
        
        # Pad to 4 lanes if needed
        while len(queue_features) < 4:
            queue_features.append(0.0)
        features.extend(queue_features[:4])
        
        # 2. Vehicle detection confidence per lane
        if sensor_frame.vehicle_detections:
            detection_confidence = np.mean([d['confidence'] for d in sensor_frame.vehicle_detections])
            detection_features = [detection_confidence] * 4
        else:
            detection_features = [0.0] * 4
        features.extend(detection_features)
        
        # 3. LiDAR occupancy grid (8x8 top-down)
        lidar_occupancy = self._lidar_to_occupancy_grid(sensor_frame.lidar_points)
        features.extend(lidar_occupancy.flatten()[:64])  # Use first 64 cells
        
        # 4. Thermal hotspots (4 quadrants)
        thermal_features = self._extract_thermal_features(sensor_frame.thermal_image)
        features.extend(thermal_features)
        
        # 5. Motion vectors (mean velocity per direction)
        motion_features = []
        if len(sensor_frame.motion_vectors) > 0:
            for direction in range(4):
                motion_features.append(np.mean(sensor_frame.motion_vectors[:, direction % 2]))
        else:
            motion_features = [0.0] * 4
        features.extend(motion_features)
        
        # 6. Weather state
        weather_features = [
            sensor_frame.weather_state['ambient_temp'] / 50,  # Normalize
            sensor_frame.weather_state['precipitation'] / 100,
            sensor_frame.weather_state['visibility'] / 1000,
            sensor_frame.weather_state['wind_speed'] / 50,
            sensor_frame.weather_state['road_wetness']
        ]
        features.extend(weather_features)
        
        # 7. Time features
        time_features = [
            sensor_frame.time_of_day / 24,
            (self.step_count % 168) / 168,  # Day of week (0-7 days)
            (self.step_count % 1440) / 1440  # Time of day cyclic
        ]
        features.extend(time_features)
        
        # 8. Pressure (wait times)
        wait_time_features = []
        for lane_id in sorted(intersection_state[self.intersection_id]['wait_times'].keys()):
            wait_time = intersection_state[self.intersection_id]['wait_times'][lane_id]
            wait_time_features.append(min(wait_time / 120, 1.0))  # Normalize to 120s max
        
        while len(wait_time_features) < 4:
            wait_time_features.append(0.0)
        features.extend(wait_time_features[:4])
        
        # Pad to 92 if needed
        while len(features) < 92:
            features.append(0.0)
        
        return np.array(features[:92], dtype=np.float32)
    
    def _lidar_to_occupancy_grid(self, 
                                 lidar_points: np.ndarray,
                                 grid_size: int = 8,
                                 range_m: float = 50.0) -> np.ndarray:
        """
        Convert LiDAR point cloud to top-down occupancy grid.
        """
        grid = np.zeros((grid_size, grid_size))
        
        if len(lidar_points) == 0:
            return grid
        
        # Project to XY plane and discretize
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        
        # Shift to [0, range_m]
        x_idx = ((x + range_m / 2) / range_m * grid_size).astype(int)
        y_idx = ((y + range_m / 2) / range_m * grid_size).astype(int)
        
        # Clamp indices
        valid = (x_idx >= 0) & (x_idx < grid_size) & (y_idx >= 0) & (y_idx < grid_size)
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]
        
        grid[x_idx, y_idx] = 1.0
        
        return grid
    
    def _extract_thermal_features(self, thermal_image: np.ndarray) -> List[float]:
        """
        Extract 4 features from thermal image (quadrants).
        """
        h, w = thermal_image.shape
        features = []
        
        for i in range(2):
            for j in range(2):
                quad = thermal_image[
                    i * h // 2:(i + 1) * h // 2,
                    j * w // 2:(j + 1) * w // 2
                ]
                # Normalize thermal to 0-1
                thermal_val = (np.mean(quad) - 20) / 100  # Assume 20-120C range
                features.append(np.clip(thermal_val, 0, 1))
        
        return features
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step: apply signal phase, observe, compute reward.
        """
        # Apply action to traffic light
        self.apply_signal_phase(action)
        
        # Simulate one SUMO step
        self.sumo.simulationStep()
        
        # Collect sensor data
        sensor_frame = self.dataset_gen.collect_frame(
            self.carla_world,
            self.get_sumo_state(),
            self.sensors
        )
        
        # Extract observation
        obs = self._extract_features(sensor_frame)
        
        # Compute reward
        reward = self._compute_reward(sensor_frame, action)
        
        # Check termination
        self.step_count += 1
        done = self.step_count >= self.max_episode_steps
        truncated = False
        
        info = {
            'scenario_id': sensor_frame.scenario_id,
            'weather': sensor_frame.weather_condition,
            'avg_wait_time': np.mean(list(sensor_frame.intersection_state[self.intersection_id]['wait_times'].values())),
            'total_vehicles': sensor_frame.intersection_state[self.intersection_id]['total_vehicles']
        }
        
        return obs, reward, done, truncated, info
    
    def _compute_reward(self, sensor_frame: SensorFrame, action: int) -> float:
        """
        Multi-component reward function.
        
        Components:
        1. Minimize average waiting time (primary objective)
        2. Maximize throughput (vehicles passing per step)
        3. Smooth phase transitions (minimize jerk)
        4. Environmental penalty (e.g., emissions during congestion)
        """
        intersection_state = sensor_frame.intersection_state[self.intersection_id]
        
        # 1. Waiting time penalty (dominant)
        avg_wait = np.mean(list(intersection_state['wait_times'].values()))
        wait_reward = -avg_wait / 10  # Normalize
        
        # 2. Throughput bonus (vehicles exiting)
        total_vehicles = intersection_state['total_vehicles']
        throughput_reward = -total_vehicles / 50 if total_vehicles > 10 else 0
        
        # 3. Smooth transitions (no unnecessary phase changes)
        phase_change_penalty = -1.0 if action != self.last_action else 0.0
        
        # 4. Environmental penalty (emissions higher during congestion + poor weather)
        weather_impact = sensor_frame.weather_state['precipitation'] / 100
        emission_penalty = -(avg_wait * weather_impact) / 20
        
        # Combined reward
        total_reward = (
            wait_reward * 0.6 +      # 60% weight on waiting time
            throughput_reward * 0.2 +  # 20% weight on throughput
            phase_change_penalty * 0.1 + # 10% on smooth transitions
            emission_penalty * 0.1    # 10% on environmental
        )
        
        self.last_action = action
        
        return float(total_reward)
```

### 5.2 Training Script

```python
def train_multi_sensor_agent(config: Dict):
    """
    End-to-end training pipeline.
    """
    # Scenario configurations to sample
    scenarios = [
        {'weather': 'clear', 'traffic': 'high', 'time_of_day': 12},
        {'weather': 'rain', 'traffic': 'high', 'time_of_day': 18},
        {'weather': 'snow', 'traffic': 'medium', 'time_of_day': 8},
        {'weather': 'fog', 'traffic': 'medium', 'time_of_day': 6},
        {'weather': 'clear', 'traffic': 'low', 'time_of_day': 22},
    ]
    
    env = MultiSensorTrafficEnv(
        config['sumo_net_file'],
        config['intersection_id'],
        max_episode_steps=1000
    )
    
    # Wrap env for stability
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    # Initialize agent
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    
    # Train with curriculum (progressively harder scenarios)
    total_timesteps = 1_000_000
    checkpoint_interval = 50_000
    
    for step in range(0, total_timesteps, checkpoint_interval):
        # Vary scenario difficulty
        scenario = scenarios[step // checkpoint_interval % len(scenarios)]
        print(f"\n=== Training Phase {step // checkpoint_interval}: {scenario} ===")
        
        model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False
        )
        
        # Save checkpoint
        model.save(f"models/agent_step_{step}")
        
        # Evaluate
        evaluate_agent(model, scenarios, num_eval_episodes=10)

def evaluate_agent(model, scenarios: List[Dict], num_eval_episodes: int = 10):
    """
    Evaluate trained agent on diverse scenarios.
    """
    results = {}
    
    for scenario in scenarios:
        episode_rewards = []
        episode_wait_times = []
        
        for episode in range(num_eval_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            wait_times = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                wait_times.append(info['avg_wait_time'])
            
            episode_rewards.append(total_reward)
            episode_wait_times.append(np.mean(wait_times))
        
        results[str(scenario)] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_wait_time': np.mean(episode_wait_times),
            'std_wait_time': np.std(episode_wait_times)
        }
    
    print("\n=== Evaluation Results ===")
    for scenario, metrics in results.items():
        print(f"\n{scenario}")
        print(f"  Avg Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Avg Wait Time: {metrics['mean_wait_time']:.1f}s ± {metrics['std_wait_time']:.1f}s")
    
    return results
```

---

## Part 6: Advanced Sensor Fusion & Pattern Discovery

### 6.1 Multi-Sensor Fusion Network

```python
import torch
import torch.nn as nn

class MultiSensorFusionNet(nn.Module):
    """
    Deep neural network for sensor fusion.
    Learns to combine RGB, thermal, LiDAR for robust decisions.
    """
    
    def __init__(self, num_actions: int = 4):
        super().__init__()
        
        # RGB stream
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        
        # Thermal stream
        self.thermal_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 64)
        )
        
        # LiDAR occupancy stream
        self.lidar_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Task heads
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, 
                lidar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, 480, 640)
            thermal: (B, 1, 480, 640)
            lidar: (B, 64)
        
        Returns:
            policy logits: (B, num_actions)
            value: (B, 1)
        """
        # Encode streams
        rgb_feat = self.rgb_encoder(rgb)
        thermal_feat = self.thermal_encoder(thermal)
        lidar_feat = self.lidar_encoder(lidar)
        
        # Fuse
        fused = torch.cat([rgb_feat, thermal_feat, lidar_feat], dim=1)
        fusion_out = self.fusion(fused)
        
        # Task heads
        policy_logits = self.policy_head(fusion_out)
        value = self.value_head(fusion_out)
        
        return policy_logits, value
```

### 6.2 Anomaly & Pattern Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

class PatternDiscovery:
    """
    Discovers new traffic patterns from synthetic sensor data.
    Identifies anomalies (glare-induced congestion, weather impacts, etc.)
    """
    
    def __init__(self, n_components: int = 20):
        self.pca = PCA(n_components=n_components)
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.patterns = []
        
    def fit(self, dataset: np.ndarray):
        """
        Learn patterns from historical data.
        
        Args:
            dataset: (N_samples, N_features) feature matrix
        """
        # Standardize
        self.mean = np.mean(dataset, axis=0)
        self.std = np.std(dataset, axis=0) + 1e-8
        dataset_norm = (dataset - self.mean) / self.std
        
        # PCA for dimensionality reduction
        dataset_pca = self.pca.fit_transform(dataset_norm)
        
        # Anomaly detection
        self.anomaly_detector.fit(dataset_pca)
    
    def detect_weather_impact_patterns(self, 
                                       training_frames: List[SensorFrame]) -> Dict:
        """
        Analyze how weather affects traffic patterns.
        
        Returns:
            Dictionary of pattern insights:
            - glare_congestion_correlation
            - rain_throughput_reduction
            - snow_queue_formation_risk
            - etc.
        """
        patterns = {}
        
        # Group by weather condition
        weather_groups = {}
        for frame in training_frames:
            weather = frame.weather_condition
            if weather not in weather_groups:
                weather_groups[weather] = []
            weather_groups[weather].append(frame)
        
        # Analyze each weather condition
        baseline_wait_time = np.mean([
            f.intersection_state[list(f.intersection_state.keys())[0]]['wait_times']
            for f in training_frames
        ])
        
        for weather, frames in weather_groups.items():
            avg_wait = np.mean([
                f.intersection_state[list(f.intersection_state.keys())[0]]['wait_times']
                for f in frames
            ])
            
            patterns[f'{weather}_wait_increase'] = (avg_wait - baseline_wait_time) / baseline_wait_time
        
        # Analyze glare impact (high sun, low visibility on detection)
        glare_frames = [f for f in training_frames if f.sun_position[0] > 30]  # High sun angle
        if glare_frames:
            avg_glare_detections = np.mean([
                len(f.vehicle_detections) for f in glare_frames
            ])
            non_glare_detections = np.mean([
                len(f.vehicle_detections) for f in training_frames 
                if f.sun_position[0] <= 30
            ])
            
            patterns['glare_detection_reduction'] = (
                (non_glare_detections - avg_glare_detections) / max(non_glare_detections, 1)
            )
        
        return patterns
    
    def suggest_phase_adaptations(self, patterns: Dict) -> List[str]:
        """
        Based on discovered patterns, suggest TLC phase adaptations.
        """
        suggestions = []
        
        if patterns.get('rain_throughput_reduction', 0) > 0.15:
            suggestions.append(
                "High rain impact detected: Extend green phases to compensate for reduced throughput"
            )
        
        if patterns.get('snow_queue_formation_risk', 0) > 0.2:
            suggestions.append(
                "Snow/ice conditions detected: Increase cycle length to reduce aggressive maneuvers"
            )
        
        if patterns.get('glare_detection_reduction', 0) > 0.1:
            suggestions.append(
                "Glare reduces detection reliability: Implement LiDAR-primary control during high sun"
            )
        
        return suggestions
```

---

## Part 7: Deployment & Iteration Strategy

### 7.1 Continuous Learning Pipeline

```
┌────────────────────────────────────────┐
│  Data Collection Phase (Simulation)    │
│  - Multi-weather scenarios             │
│  - Time-of-day variations              │
│  - Traffic demand patterns             │
└──────────┬───────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  Dataset Generation                    │
│  - Sensor fusion                       │
│  - Environmental effects               │
│  - Annotation & metadata               │
└──────────┬───────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  Pattern Discovery                     │
│  - Anomaly detection                   │
│  - Weather correlations                │
│  - Failure mode identification         │
└──────────┬───────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  RL Agent Training                     │
│  - Multi-sensor fusion network         │
│  - Curriculum learning                 │
│  - Evaluation on diverse scenarios     │
└──────────┬───────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│  Real-World Testing (Field Deployment) │
│  - Gradual rollout                     │
│  - A/B testing vs. baseline            │
│  - Feedback loop                       │
└──────────┬───────────────────────────┘
           │
           ▼ (Failures, gaps detected)
      [Loop Back to Data Collection]
```

### 7.2 Key Metrics to Track

```python
class MetricsTracker:
    """
    Track training progress and real-world performance.
    """
    
    # Simulation Metrics
    avg_wait_time: float                    # seconds
    throughput: float                       # vehicles/hour
    queue_length_variance: float            # vehicles
    phase_efficiency: float                 # 0-1 (phases with traffic)
    
    # Environmental Adaptation
    weather_robustness: Dict[str, float]    # Performance in each weather
    time_of_day_consistency: float          # Performance variation across 24h
    glare_resilience: float                 # Detection accuracy in glare
    
    # Agent Learning
    training_reward_trend: List[float]      # Increasing over time?
    convergence_speed: float                # Timesteps to 90% performance
    generalization_gap: float               # Train vs. test performance
    
    # Real-World (if deployed)
    actual_wait_time_reduction: float       # % vs. baseline
    safety_incidents: int
    pedestrian_compliance: float            # Detection/tracking accuracy
    weather_failure_rate: float             # Dropouts during poor conditions
```

---

## Part 8: Additional Sensor Modalities (Future Extensions)

### Sensors to Add Later

1. **Radar**: Long-range detection, motion estimation
   - Doppler processing for velocity
   - Clutter rejection algorithms
   - Fusion with optical flow

2. **Acoustic Sensors**: Emergency vehicle sirens, noise-based congestion
   - Siren detection & localization
   - Frequency-domain analysis

3. **V2X (Vehicle-to-Infrastructure)**: Direct vehicle data feeds
   - Brake signal broadcasting
   - Intended path announcements
   - Cooperative adaptive cruise control (CACC)

4. **Magnetic Induction Loops**: Buried sensor counts
   - Presence detection per lane
   - Occupancy duration

5. **Edge Computing Integration**: On-device inference
   - Latency-aware control
   - Bandwidth constraints

---

## Summary: Implementation Roadmap

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| **1. Setup** | Week 1-2 | CARLA-SUMO bridge, sensor infrastructure |
| **2. Sensor Simulation** | Week 2-4 | LiDAR, thermal, motion detection working |
| **3. Environmental Effects** | Week 4-5 | Glare, rain, snow, visibility modules |
| **4. Dataset Generation** | Week 5-6 | 50k+ multi-modal frames across scenarios |
| **5. RL Training** | Week 6-8 | Agent converges, pattern discovery working |
| **6. Evaluation & Iteration** | Week 8-10 | A/B testing, failure analysis |
| **7. Deployment** | Week 10+ | Staged rollout with monitoring |

---

## References & Resources

- CARLA Simulator: https://carla.org/
- SUMO Simulator: https://sumo.dlr.de/
- sumo-rl Library: https://github.com/lcodeca/sumo-rl
- YOLOv5 Detection: https://github.com/ultralytics/yolov5
- Stable Baselines3: https://stable-baselines3.readthedocs.io/

---

**This framework positions your agent as a true intersection singularity—synthesizing patterns across multiple modalities that traditional systems can't capture. The environmental layer (glare, weather, road conditions) creates edge cases that drive genuine improvements in robustness and efficiency.**

