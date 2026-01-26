import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorFusionDQN(nn.Module):
    """
    Attention-based Multi-Branch Network for fusing different sensor modalities.
    Dynamically weighs sensor inputs based on context (weather, reliability).
    
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
        self.output_dim = action_dim 
        
        # --- Encoders ---
        
        # 1. Ground Truth Encoder (FC)
        self.gt_encoder = nn.Sequential(
            nn.Linear(60, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # 2. LiDAR Encoder (Spatial approx with FC)
        self.lidar_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # 3. Context Encoder (IR, Motion, Weather, Time)
        # 5 + 7 + 4 + 2 = 18 features
        self.context_encoder = nn.Sequential(
            nn.Linear(18, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )
        
        # --- Attention Mechanism ---
        # Computes importance weights for [GT, LiDAR, Context]
        # Input: concatenation of all embeddings (64*3 = 192)
        self.attention_net = nn.Sequential(
            nn.Linear(192, 64),
            nn.Tanh(),
            nn.Linear(64, 3), # 3 Heads: GT, LiDAR, Context
            nn.Softmax(dim=1)
        )
        
        # --- Fusion Layer ---
        self.fusion_layer = nn.Sequential(
            nn.Linear(192, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
        # --- Dueling Heads ---
        
        # Value Stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, state):
        batch_size = state.size(0)
        
        # Slicing the input vector
        gt_input = state[:, :60]
        lidar_input = state[:, 60:124]
        # Context features (IR, Motion, Weather, Time)
        context_input = state[:, 124:142]
        
        # Encode
        gt_emb = self.gt_encoder(gt_input)        # [B, 64]
        lidar_emb = self.lidar_encoder(lidar_input) # [B, 64]
        context_emb = self.context_encoder(context_input) # [B, 64]
        
        # Stack embeddings
        # [B, 192]
        combined_emb = torch.cat([gt_emb, lidar_emb, context_emb], dim=1)
        
        # Compute Attention Weights
        # [B, 3] -> (w_gt, w_lidar, w_context)
        attn_weights = self.attention_net(combined_emb)
        
        # Apply Attention
        # Reshape for broadcasting: weights [B, 3, 1] * stack [B, 3, 64]
        emb_stack = torch.stack([gt_emb, lidar_emb, context_emb], dim=1) # [B, 3, 64]
        attn_expanded = attn_weights.unsqueeze(2) # [B, 3, 1]
        
        # Weighted sum? Or Concatenation?
        # Weighted concatenation is conceptually cleaner for preserving unique info
        # Let's do element-wise scaling then flatten
        weighted_emb = emb_stack * attn_expanded # [B, 3, 64]
        fused_input = weighted_emb.view(batch_size, -1) # [B, 192]
        
        # Deep Fusion
        fused = self.fusion_layer(fused_input)
        
        # Dueling Output
        val = self.value_stream(fused)
        adv = self.advantage_stream(fused)
        
        # Q = V + (A - mean(A))
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_val
