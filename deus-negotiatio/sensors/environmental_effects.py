import numpy as np
import random

class EnvironmentalEffects:
    """
    Simulates weather and environmental conditions.
    Degrades sensor performance based on current state.
    """
    
    TYPES = ['clear', 'rain', 'snow', 'fog', 'glare']
    
    def __init__(self):
        self.current_state = {
            'type': 'clear',
            'severity': 0.0, # 0.0 to 1.0
            'precipitation': 0.0,
            'visibility': 1.0, # 1.0 = perfect, 0.0 = blind
            'surface_friction': 1.0,
            'glare_intensity': 0.0
        }
        self.step_counter = 0
        
    def step(self):
        """Update weather state (dynamic evolution)."""
        self.step_counter += 1
        
        # Slowly evolve weather
        # For training, we might force specific scenarios, 
        # but here we allow drift
        if self.step_counter % 1000 == 0:
            if random.random() < 0.2: # 20% chance to change weather loop
                self._change_weather()
                
    def _change_weather(self):
        new_type = random.choice(self.TYPES)
        severity = random.uniform(0.3, 1.0) if new_type != 'clear' else 0.0
        
        self.current_state['type'] = new_type
        self.current_state['severity'] = severity
        
        # Effect logic
        if new_type == 'clear':
            self.current_state.update({
                'precipitation': 0.0, 'visibility': 1.0, 
                'surface_friction': 1.0, 'glare_intensity': 0.0
            })
        elif new_type == 'rain':
            self.current_state.update({
                'precipitation': severity, 
                'visibility': 1.0 - (severity * 0.4), 
                'surface_friction': 1.0 - (severity * 0.3),
                'glare_intensity': 0.0
            })
        elif new_type == 'snow':
            self.current_state.update({
                'precipitation': severity, 
                'visibility': 1.0 - (severity * 0.7), 
                'surface_friction': 1.0 - (severity * 0.6),
                'glare_intensity': 0.0
            })
        elif new_type == 'fog':
            self.current_state.update({
                'precipitation': 0.1, 
                'visibility': 1.0 - (severity * 0.9), 
                'surface_friction': 0.9,
                'glare_intensity': 0.0
            })
        elif new_type == 'glare':
            # Sun glare: high visibility generally, but blinding in specific angles
            self.current_state.update({
                'precipitation': 0.0, 
                'visibility': 1.0, 
                'surface_friction': 1.0,
                'glare_intensity': severity
            })

    def apply_to_lidar(self, lidar_data):
        """Degrade LiDAR data based on weather."""
        severity = self.current_state['precipitation']
        
        if severity > 0:
            # Rain/Snow adds noise and reduces range
            # Range reduction
            valid_mask = np.linalg.norm(lidar_data['points'][:, :2], axis=1) < (100.0 * (1 - severity * 0.5))
            lidar_data['points'] = lidar_data['points'][valid_mask]
            
            # Add ghost points (false positives from particles)
            if random.random() < severity:
                num_ghosts = int(severity * 50)
                ghosts = np.random.uniform(-50, 50, (num_ghosts, 3))
                lidar_data['points'] = np.vstack([lidar_data['points'], ghosts])
                
        return lidar_data

    def apply_to_motion(self, motion_data):
        """Degrade camera motion detection."""
        # Glare or Fog kills optical flow reliability
        degradation = 0.0
        if self.current_state['type'] == 'fog':
            degradation = self.current_state['severity']
        elif self.current_state['type'] == 'glare':
            degradation = self.current_state['severity'] * 0.8
            
        if degradation > 0:
            # Add noise to vectors
            noise = np.random.normal(0, degradation, 4)
            motion_data['motion_vectors'] = [
                v + n for v, n in zip(motion_data['motion_vectors'], noise)
            ]
            
            # Reduce confidence/zero out if too bad
            if degradation > 0.8:
                motion_data['motion_vectors'] = [0.0] * 4
                
        return motion_data

    def get_state_feature(self):
        """Return normalized weather feature vector."""
        # [Precip, Vis, Friction, Glare]
        return [
            self.current_state['precipitation'],
            self.current_state['visibility'],
            self.current_state['surface_friction'],
            self.current_state['glare_intensity']
        ]
