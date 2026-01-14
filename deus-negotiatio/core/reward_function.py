class RewardFunction:
    """
    Computes the multi-objective reward signal.
    Objectives:
    - minimize_queue
    - minimize_delay
    - pedestrian_safety
    - minimize_emissions (approx. by stopped vehicles)
    """
    def __init__(self, weights=None):
        # Default weights from architecture doc
        self.weights = weights or {
            'vehicle': 0.4,
            'pedestrian': 0.3,
            'emissions': 0.2,
            'waiting': 0.1
        }
    
    def compute_reward(self, prev_state, action, current_state, metrics):
        """
        metrics: dict containing raw measurements
        - queues: total queue length
        - ped_wait_max: max pedestrian wait time in seconds
        - stopped_vehicles: count of stopped vehicles
        - avg_wait: average vehicle wait time
        - collision_risk: 0 or 1 (boolean or prob)
        """
        
        # 1. Vehicle Efficiency (Minimize Queues)
        # Reward is negative of cost
        r_vehicle = -1.0 * sum(metrics.get('queues', []))
        
        # 2. Pedestrian Safety
        # Penalize hard if max wait exceeds threshold or collision risk
        ped_wait = metrics.get('ped_wait_max', 0)
        collision = metrics.get('collision_risk', 0)
        
        # Penalty increases non-linearly with wait time
        r_pedestrian = -1.0 * (ped_wait / 30.0) 
        if collision > 0:
            r_pedestrian -= 10.0 # Hard penalty for safety violations
            
        # 3. Emissions (Stopped Vehicles)
        r_emissions = -1.0 * metrics.get('stopped_vehicles', 0)
        
        # 4. Waiting Time (Cumulative Delay)
        r_waiting = -1.0 * metrics.get('avg_wait', 0)
        
        # Weighted Sum
        total_reward = (
            self.weights['vehicle'] * r_vehicle +
            self.weights['pedestrian'] * r_pedestrian +
            self.weights['emissions'] * r_emissions +
            self.weights['waiting'] * r_waiting
        )
        
        return total_reward
