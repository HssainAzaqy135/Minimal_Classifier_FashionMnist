import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time

class GatingNetwork(nn.Module):
    def __init__(self, input_dim,num_experts,gating_net):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(input_dim * input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, num_experts)
        # )
        self.net = gating_net
        self.proj_to_experts = nn.Linear(10, num_experts)
        
    def forward(self, x):
        logits = self.net(x)
        expert_scores = self.proj_to_experts(logits)
        return F.softmax(expert_scores, dim=1)  # [batch_size, num_experts]

class MOE_CNN(nn.Module):
    def __init__(self, experts, input_dim,gating_net):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.gate = GatingNetwork(input_dim=input_dim, num_experts=self.num_experts,gating_net=gating_net)

    def forward(self, x):
        # Get gating weights
        gate_weights = self.gate(x)  # [B, E]

        # Run input through each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [B, E, C]

        # Weighted sum of expert outputs
        gate_weights = gate_weights.unsqueeze(-1)  # [B, E, 1]
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # [B, C]
        return output
        
    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
