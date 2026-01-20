import numpy as np
from sklearn.cluster import DBSCAN
import time

class LidarSimulator:
    """
    Simulates a 64-channel LiDAR sensor based on SUMO vehicle positions.
    Generates point clouds and occupancy grids.
    """
    
    def __init__(self, 
                 max_range=100.0,
                 grid_size=8,
                 points_per_vehicle=50,
                 noise_stddev=0.05):
        self.max_range = max_range
        self.grid_size = grid_size
        self.points_per_vehicle = points_per_vehicle
        self.noise_stddev = noise_stddev
        
    def scan(self, vehicles):
        """
        Simulate a LiDAR scan of the current vehicles.
        
        Args:
            vehicles (list): List of vehicle dictionaries from SUMO
                           [{'id': str, 'x': float, 'y': float, ...}]
                           Note: coordinates should be relative to sensor (intersection center)
        
        Returns:
            dict: {
                'points': np.array (N, 3),
                'occupancy_grid': np.array (grid_size, grid_size),
                'clusters': int (number of detected objects)
            }
        """
        all_points = []
        
        for veh in vehicles:
            # Generate points for vehicle body (simplified as a box)
            # Assuming average car size 4.5m x 1.8m x 1.5m
            length = 4.5
            width = 1.8
            height = 1.5
            
            # Vehicle center (already relative to intersection)
            cx, cy = veh['x'], veh['y']
            
            # Generate random points within vehicle volume
            # In a real LiDAR, we'd only see the surface facing the sensor
            # Simplification: uniformly sample the volume then filter by visibility
            
            points = np.random.uniform(
                low=[cx - length/2, cy - width/2, 0],
                high=[cx + length/2, cy + width/2, height],
                size=(self.points_per_vehicle, 3)
            )
            
            # Add sensor noise
            noise = np.random.normal(0, self.noise_stddev, points.shape)
            points += noise
            
            all_points.append(points)
            
        if not all_points:
            return {
                'points': np.empty((0, 3)),
                'occupancy_grid': np.zeros((self.grid_size, self.grid_size)),
                'clusters': 0
            }
            
        # Combine all points
        cloud = np.vstack(all_points)
        
        # Filter by max range
        distances = np.linalg.norm(cloud[:, :2], axis=1) # 2D distance
        cloud = cloud[distances <= self.max_range]
        
        if len(cloud) == 0:
             return {
                'points': np.empty((0, 3)),
                'occupancy_grid': np.zeros((self.grid_size, self.grid_size)),
                'clusters': 0
            }

        # Beaming Occupancy Grid (Top-down)
        grid = self._points_to_grid(cloud)
        
        # Object Detection (Clustering)
        # Using DBSCAN to find clusters of points (representing vehicles)
        # eps=1.0m (vehicles closer than 1m merge), min_samples=5 points
        clustering = DBSCAN(eps=1.5, min_samples=5).fit(cloud)
        num_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        return {
            'points': cloud,
            'occupancy_grid': grid,
            'clusters': num_clusters
        }
        
    def _points_to_grid(self, points):
        """Convert point cloud to top-down 2D occupancy grid."""
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Map X,Y to grid indices
        # Range: [-max_range, max_range] -> [0, grid_size]
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Normalized coordinates 0..1
        norm_x = (x + self.max_range) / (2 * self.max_range)
        norm_y = (y + self.max_range) / (2 * self.max_range)
        
        # Grid indices
        idx_x = (norm_x * self.grid_size).astype(int)
        idx_y = (norm_y * self.grid_size).astype(int)
        
        # Clip to valid range
        valid_mask = (idx_x >= 0) & (idx_x < self.grid_size) & \
                     (idx_y >= 0) & (idx_y < self.grid_size)
                     
        idx_x = idx_x[valid_mask]
        idx_y = idx_y[valid_mask]
        
        # Fill grid (occupancy count)
        # Flatten indices to use bincount
        flat_indices = idx_x * self.grid_size + idx_y
        counts = np.bincount(flat_indices, minlength=self.grid_size**2)
        
        grid = counts.reshape(self.grid_size, self.grid_size)
        
        # Normalize/Cap occupancy (binary or density)
        # For RL, density is often better. 
        # Normalize so 1.0 means "heavy cluster" (e.g., >5 points)
        grid = np.clip(grid / 5.0, 0, 1.0)
        
        return grid
