import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorFusionDQN(nn.Module):
    """
    Multi-Branch Network for fusing different sensor modalities.
    
    Inputs (concatenated 156-dim vector):
    - 0-59:    SUMO Ground Truth (60)
    - 60-123:  LiDAR Occupancy (64) -> Reshaped to 8x8
    - 124-128: Infrared (5)
    - 129-135: Motion (7)
    - 136-139: Weather (4)
    - 140-141: Time (2)
    - 142-155: Padding (14)
    """
    
    def __init__(self, state_dim, action_dim):
        super(SensorFusionDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = action_dim # Required for agent's select_action
        
        # --- Encoders ---
        
        # 1. Ground Truth Encoder (FC)
        self.gt_encoder = nn.Sequential(
            nn.Linear(60, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 2. LiDAR Encoder (Spatial)
        # Represents 8x8 grid. We can use a small CNN or just FC since it's small.
        # Let's use FC for simplicity but keep it separate to learn spatial patterns.
        self.lidar_encoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 3. Context Encoder (IR, Motion, Weather, Time)
        # 5 + 7 + 4 + 2 = 18 features
        self.context_encoder = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # --- Fusion Layer ---
        # Concatenate outputs: 32 + 32 + 16 = 80 features
        self.fusion_layer = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Dropout(0.1) # Robustness
        )
        
        # --- Dueling Heads ---
        
        # Value Stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage Stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state):
        # Slicing the input vector
        # Assuming state is [batch_size, 156]
        
        gt_input = state[:, :60]
        lidar_input = state[:, 60:124]
        
        # Context features are scattered in our env construction but let's assume contiguous indices for encoder simplicity
        # IR(5) + Motion(7) + Weather(4) + Time(2) = 124 to 142
        context_input = state[:, 124:142]
        
        # Encode
        gt_feat = self.gt_encoder(gt_input)
        lidar_feat = self.lidar_encoder(lidar_input)
        context_feat = self.context_encoder(context_input)
        
        # Fuse
        combined = torch.cat([gt_feat, lidar_feat, context_feat], dim=1)
        fused = self.fusion_layer(combined)
        
        # Dueling Output
        val = self.value_stream(fused)
        adv = self.advantage_stream(fused)
        
        # Q = V + (A - mean(A))
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_val
