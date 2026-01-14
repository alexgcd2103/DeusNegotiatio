import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture.
    Splits the network into Value (V) and Advantage (A) streams.
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], dropout=0.2):
        super(DuelingDQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature Extraction Layer (Common)
        # Using simple Dense layers here, but Architecture doc mentions CNN if raw spatial data used.
        # Since StateEncoder outputs a vector, we stick to MLP for now.
        layers = []
        in_size = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_size, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_size = h_dim
            
        self.feature_layer = nn.Sequential(*layers)
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine V and A
        # q = v + (a - mean(a))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_vals
