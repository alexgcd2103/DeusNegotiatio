import torch
import torch.nn as nn
import torch.nn.functional as F
from core.noisy_linear import NoisyLinear

class SensorFusionDQN(nn.Module):
    """
    Attention-based Multi-Branch Network for fusing different sensor modalities.
    Dynamically weighs sensor inputs based on context (weather, reliability).
    
    Inputs (concatenated 172-dim vector):
    - 0-47:    SUMO Ground Truth (48: 16 lanes * 3)
    - 48-53:   Phase Encoding (6)
    - 54-117:  LiDAR Occupancy (64)
    - 118-122: Infrared (5)
    - 123-129: Motion (7)
    - 130-133: Weather (4)
    - 134-135: Time (2)
    - 136-151: TiQ (16: 16 lanes * 1)
    - 152-171: Padding (20)
    """
    
    def __init__(self, state_dim, action_dim):
        super(SensorFusionDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = action_dim 
        
        # --- Encoders ---
        
        # 1. Ground Truth Encoder (FC)
        # 48 (SUMO) + 6 (Phase) + 16 (TiQ) = 70 features
        self.gt_encoder = nn.Sequential(
            nn.Linear(70, 128),
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
            NoisyLinear(128, 1)
        )
        
        # Advantage Stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            NoisyLinear(128, action_dim)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
        
    def forward(self, state):
        batch_size = state.size(0)
        
        # Slicing the input vector
        # GT: SUMO (48) + Phase (6) + TiQ (16)
        # We need to gather these from their specific indices
        sumo_phase = state[:, :54]      # 0-53
        tiq = state[:, 136:152]         # 136-151
        gt_input = torch.cat([sumo_phase, tiq], dim=1) # 70
        
        lidar_input = state[:, 54:118]   # 54-117
        
        # Context features (IR, Motion, Weather, Time)
        context_input = state[:, 118:136] # 118-135
        
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

    def reset_noise(self):
        """Resets the noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
