# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:23:41 2026

@author: Santosh Prakash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=0,
            dilation=dilation,
            **kwargs
        )
        
    def forward(self, x):
        if self.padding != 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = CausalConv1d(self.d_inner, self.d_inner, self.d_conv)
        
        # SSM parameters
        #self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, self.d_state)))
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Selective parameters
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # Project input
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv step
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # Discretization
        dt = self.dt_proj(x)
        dt = torch.sigmoid(dt)
        dt = torch.clamp(dt, 1e-4, 1.0)
        
        A = -torch.exp(self.A_log.float())
        B = self.B_proj(x)
        C = self.C_proj(x)
        
        # Selective scan
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        # Gating and output
        y = y * F.silu(z)
        return self.out_proj(y)
    
    def selective_scan(self, u, delta, A, B, C, D):
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[-1]
        
        # Proper broadcasting shapes
        delta = delta.unsqueeze(-1)  # (b, l, d_inner, 1)
        A = A.view(1, 1, d_inner, d_state)  # (1, 1, d_inner, d_state)
        B = B.unsqueeze(2)  # (b, l, 1, d_state)
        C = C.unsqueeze(2)  # (b, l, 1, d_state)
        
        # Discretize
        #deltaA = torch.exp(delta * A)
        deltaA = torch.exp(torch.clamp(delta * A, -5, 5))
        deltaB = delta * B * u.unsqueeze(-1)  # (b, l, d_inner, d_state)
        
        # Initialize state
        state = torch.zeros(batch, d_inner, d_state, device=u.device)
        outputs = []
        
        # Recurrent scan
        for i in range(seq_len):
            #state = deltaA[:, i] * state + deltaB[:, i]
            state = deltaA[:, i] * state + deltaB[:, i] + 1e-6
            output = torch.einsum('bdn,bdn->bd', state, C[:, i])
            outputs.append(output + D * u[:, i])
        
        return torch.stack(outputs, dim=1)

class DirectionalScan(nn.Module):
    def __init__(self, ssm_layer, d_model):
        super().__init__()
        self.ssm_layer = ssm_layer
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, h, w):
        b, l, d = x.shape
        assert l == h * w
        
        # Horizontal scans
        x_h = rearrange(x, 'b (h w) d -> (b w) h d', h=h, w=w)
        x_h = self.ssm_layer(x_h)
        x_h = rearrange(x_h, '(b w) h d -> b (h w) d', h=h, w=w)
        
        # Vertical scans
        x_v = rearrange(x, 'b (h w) d -> (b h) w d', h=h, w=w)
        x_v = self.ssm_layer(x_v)
        x_v = rearrange(x_v, '(b h) w d -> b (h w) d', h=h, w=w)
        
        return self.proj(x_h + x_v)

class VisionMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.scan = DirectionalScan(self.ssm, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, h, w):
        #x = x + self.scan(self.norm(x), h, w)
        x_norm = self.norm(x)
        #x_norm = torch.clamp(x_norm, -5, 5)
        x = x + self.scan(self.norm(x), h, w)
        x = x + self.scan(x_norm, h, w)
        x = x + self.mlp(self.norm(x))
        return x

