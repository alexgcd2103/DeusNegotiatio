import numpy as np

class StateEncoder:
    """
    Encodes raw traffic data into a state vector for the DRL agent.
    State components:
    - Queue lengths (normalized)
    - Average velocities (normalized)
    - Current phase (one-hot)
    - Time remaining in phase (normalized)
    - Pedestrian waiting count (normalized)
    """
    def __init__(self, config):
        self.config = config
        self.max_queue = 50.0  # Max vehicles per lane for normalization
        self.max_speed = 15.0  # Max speed in m/s
        self.num_phases = 8    # Standard NEMA phases

    def encode(self, traffic_data):
        """
        Args:
            traffic_data (dict): Raw data from SURTRAC detector service.
                {
                    'queues': [q1, q2, ...],
                    'velocities': [v1, v2, ...],
                    'current_phase': int,
                    'phase_time_remaining': float,
                    'pedestrian_counts': [p1, p2, ...]
                }
        Returns:
            np.array: Flattened state vector
        """
        # 1. Normalize Queue Lengths
        queues = np.array(traffic_data.get('queues', [])) / self.max_queue
        queues = np.clip(queues, 0, 1)

        # 2. Normalize Velocities
        velocities = np.array(traffic_data.get('velocities', [])) / self.max_speed
        velocities = np.clip(velocities, 0, 1)

        # 3. One-hot encode phase
        phase = traffic_data.get('current_phase', 0)
        phase_one_hot = np.zeros(self.num_phases)
        if 0 <= phase < self.num_phases:
            phase_one_hot[phase] = 1.0

        # 4. Normalize Time Remaining
        time_rem = traffic_data.get('phase_time_remaining', 0.0) / 60.0 # Normalizing by estimated max cycle
        time_rem = np.clip(time_rem, 0, 1)
        
        # 5. Pedestrian features
        peds = np.array(traffic_data.get('pedestrian_counts', [])) / 10.0 # Normalize by 10
        peds = np.clip(peds, 0, 1)

        # Concatenate all features
        state = np.concatenate([
            queues,
            velocities,
            phase_one_hot,
            [time_rem],
            peds
        ])
        
        return state.astype(np.float32)

    def get_state_dim(self):
        # Helper to calculate dimension based on a dummy input or config
        # This implementation assumes fixed sizes for simplicity, 
        # normally this would be dynamic based on intersection topology
        # Mocking generic 4-way intersection: 4 approaches * lanes
        # For now, let's assume specific dimension logic or return a placeholder
        # In a real impl, this depends on the specific intersection config
        return 32 # Placeholder dimension
