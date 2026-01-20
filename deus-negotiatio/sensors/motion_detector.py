import numpy as np

class MotionDetector:
    """
    Simulates a smart camera with motion detection and object classification.
    """
    
    def __init__(self, range_m=60.0):
        self.range_m = range_m
        
    def detect(self, vehicles):
        """
        Analyze motion vectors and class distribution.
        
        Returns:
            dict: {
                'motion_vectors': [vx_mean, vy_mean, speed_mean, variance],
                'class_dist': [pct_cars, pct_trucks, pct_buses]
            }
        """
        if not vehicles:
            return {
                'motion_vectors': [0.0, 0.0, 0.0, 0.0],
                'class_dist': [0.0, 0.0, 0.0]
            }
            
        # Filter by range
        visible_vehs = [
            v for v in vehicles 
            if np.sqrt(v['x']**2 + v['y']**2) <= self.range_m
        ]
        
        if not visible_vehs:
             return {
                'motion_vectors': [0.0, 0.0, 0.0, 0.0],
                'class_dist': [0.0, 0.0, 0.0]
            }
            
        # 1. Motion Vectors (Optical Flow Simulation)
        # Calculate mean velocity vector of the scene
        velocities = np.array([[v.get('speed', 0) * np.cos(v.get('angle', 0)), 
                                v.get('speed', 0) * np.sin(v.get('angle', 0))] 
                               for v in visible_vehs])
        
        speeds = np.array([v.get('speed', 0) for v in visible_vehs])
        
        mean_vel = np.mean(velocities, axis=0) if len(velocities) > 0 else [0, 0]
        mean_speed = np.mean(speeds) if len(speeds) > 0 else 0
        speed_var = np.var(speeds) if len(speeds) > 0 else 0
        
        # Normalize
        # Max expected speed ~20m/s
        motion_vec = [
            np.clip(mean_vel[0] / 20.0, -1, 1),
            np.clip(mean_vel[1] / 20.0, -1, 1),
            np.clip(mean_speed / 20.0, 0, 1),
            np.clip(speed_var / 100.0, 0, 1) # Speed variance (chaos metric)
        ]
        
        # 2. Classification
        # SUMO 'vClass' or infer from length
        classes = {'car': 0, 'truck': 0, 'bus': 0}
        total = len(visible_vehs)
        
        for v in visible_vehs:
            # Simple heuristic if explicit class missing
            length = v.get('length', 5.0)
            if length < 6.0:
                classes['car'] += 1
            elif length < 10.0:
                classes['truck'] += 1
            else:
                classes['bus'] += 1
                
        class_dist = [
            classes['car'] / total,
            classes['truck'] / total,
            classes['bus'] / total
        ]
        
        return {
            'motion_vectors': motion_vec,
            'class_dist': class_dist
        }
