import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, TAGConv
import numpy as np
import sys
sys.path.append('..')
from to_graph import *

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.2
"""
Adaptive Fusion HybridLIFGNN Model
Implements multiple fusion strategies for spatial and temporal streams
"""

class EBNorm(nn.BatchNorm2d):
    """
    Event-Based Batch Normalization (tdBN).

    Normalizes input across spatial and temporal dimensions.
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(EBNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 1, 3])
            var = input.var([0, 1, 3], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = thresh * (input - mean[None, None, :, None]) / (torch.sqrt(var[None, None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :, None] + self.bias[None, None, :, None]

        return input

# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update, for GCN
def mem_update_conv(ops, x, edge_idxs, mem, spike, bn=None):
    """
    Update membrane with graph conv output; optional BN applied pre-LIF.
    """
    conv_out = ops(x, edge_idxs)
    if bn is not None:
        conv_out = apply_ebn(bn, conv_out)
    mem = mem * decay * (1. - spike) + conv_out
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem)
    return mem, spike


# cnn_layer(in_channels, out_channels)
cfg_cnn = [(2, 64),]
# kernel size
cfg_s = [39, 39]
# fc layer
cfg_fc = [128, 256]
gamma = 0.2  # dropout coefficient


def apply_ebn(bn_layer, spike_tensor):
    """
    Apply event-based batch norm to spike tensor shaped [B, N, C].
    Reshapes to [B, N, C, 1] for tdBN (channel on dim=2), then back.
    """
    b, n, c = spike_tensor.shape
    x = spike_tensor.unsqueeze(-1)  # [B, N, C, 1]
    x = bn_layer(x)
    x = x.squeeze(-1)  # [B, N, C]
    return x


class SimpleFusion(nn.Module):
    """
    Simple learnable weighted fusion
    Learns a single scalar weight to balance spatial and temporal streams
    """
    def __init__(self):
        super(SimpleFusion, self).__init__()
        # Initialize alpha to 0.5 (equal weight)
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, spatial_output, temporal_output):
        # Clamp alpha to [0, 1] for interpretability
        alpha = torch.sigmoid(self.alpha)
        output = alpha * spatial_output + (1 - alpha) * temporal_output
        return output, alpha


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism
    Uses features from both streams to compute adaptive weights
    """
    def __init__(self, num_classes):
        super(GatedFusion, self).__init__()
        # Gate network: takes concatenated features and outputs fusion weight
        self.gate = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial_output, temporal_output):
        # Concatenate both stream outputs
        combined = torch.cat([spatial_output, temporal_output], dim=1)
        # Compute fusion weight
        alpha = self.gate(combined)
        # Fuse outputs
        output = alpha * spatial_output + (1 - alpha) * temporal_output
        return output, alpha.mean()



class HybridLIFGNN_AdaptiveFusion(nn.Module):
    """
    Hybrid LIF-GNN with Adaptive Fusion
    
    Args:
        num_classes: Number of output classes
        time_length: Length of temporal sequence
        k: Number of neighbors for graph construction
        device: Computing device
        sparse: Use sparse temporal graph
        fusion_type: Type of fusion mechanism ('simple', 'gated', 'attention')
    """
    def __init__(self, num_classes=20, time_length=250, k=3, device="cuda:0", 
                 sparse=True, fusion_type='simple'):
        super(HybridLIFGNN_AdaptiveFusion, self).__init__()
        
        in_planes, out_planes = cfg_cnn[0]
        
        # Spatial and temporal graph convolutions
        self.conv1 = TAGConv(in_planes, out_planes, K=3)
        self.conv2 = TAGConv(in_planes, out_planes, K=3)
        self.s_bn = EBNorm(out_planes)
        self.t_bn = EBNorm(out_planes)

        # Spatial stream fully connected layers
        self.fc1 = nn.Linear(cfg_s[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], num_classes)

        # Temporal stream fully connected layers
        self.tfc1 = nn.Linear(time_length * cfg_cnn[-1][1], cfg_fc[0])
        self.tfc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.tfc3 = nn.Linear(cfg_fc[1], num_classes)

        self.num_classes = num_classes
        self.time_length = time_length
        self.sparse = sparse
        self.fusion_type = fusion_type
        
        # Graph constructors
        self.Sgraph = TactileSpatialGraph(k)
        self.Tgraph = TactileTemporalGraph(self.time_length, self.sparse)
        self.device = device
        
        if fusion_type == 'simple':
            self.fusion = SimpleFusion()
        # # Adaptive fusion module
        # if fusion_type == 'simple':
        #     self.fusion = SimpleFusion()
        # elif fusion_type == 'gated':
        #     self.fusion = GatedFusion(num_classes)
        # else:
        #     raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        print(f"Model initialized with {fusion_type} fusion")

    def forward(self, input, training=True):
        data = input.to(self.device)
        sizes = data.size()
        time_window = sizes[-1]
        batch_size = sizes[0]
        
        # Initialize membrane potentials and spikes for spatial stream
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_s[0], cfg_cnn[0][1], device=self.device)
        h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=self.device)
        h2_mem = h2_spike = torch.zeros(batch_size, cfg_fc[1], device=self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, self.num_classes, device=self.device)

        # Initialize membrane potentials and spikes for temporal stream
        t1_mem = t1_spike = torch.zeros(batch_size, time_window, cfg_cnn[0][1], device=self.device)
        th1_mem = th1_spike = torch.zeros(batch_size, cfg_fc[0], device=self.device)
        th2_mem = th2_spike = torch.zeros(batch_size, cfg_fc[1], device=self.device)
        th3_mem = th3_spike = th3_sumspike = torch.zeros(batch_size, self.num_classes, device=self.device)

        # ==========================
        # Spatial Stream Processing
        # ==========================
        inputs = data.split(1, dim=len(sizes) - 1)
        for step in range(time_window):
            x = inputs[step].squeeze()
            x = x.to(self.device)
            
            # Construct spatial graph
            graph_data = self.Sgraph(x)
            x = graph_data.x.to(self.device)
            edge_idxs = graph_data.edge_index.to(self.device)

            # Graph convolution with LIF neurons
            c1_mem, c1_spike = mem_update_conv(self.conv1, x, edge_idxs, c1_mem, c1_spike, bn=self.s_bn)
            x = c1_spike
            x = x.view(batch_size, -1)

            # Fully connected layers with LIF neurons
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike

        outputs_spatial = h3_sumspike / time_window

        # ==========================
        # Temporal Stream Processing
        # ==========================
        data_temporal_inputs = data
        temporal_inputs = data_temporal_inputs.split(1, dim=1)
        spatial_window = sizes[1]  # 39
        
        for temporal_step in range(spatial_window):
            temporal_x = temporal_inputs[temporal_step].squeeze()
            temporal_x = temporal_x.transpose(1, 2)
            temporal_x = temporal_x.to(self.device)
            
            # Construct temporal graph
            temporal_graph_data = self.Tgraph(temporal_x)
            temporal_x = temporal_graph_data.x.to(self.device)
            temporal_edge_idxs = temporal_graph_data.edge_index.to(self.device)
            
            # Graph convolution with LIF neurons
            t1_mem, t1_spike = mem_update_conv(self.conv2, temporal_x, temporal_edge_idxs, t1_mem, t1_spike, bn=self.t_bn)
            temporal_x = t1_spike
            temporal_x = temporal_x.view(batch_size, -1)

            # Fully connected layers with LIF neurons
            th1_mem, th1_spike = mem_update(self.tfc1, temporal_x, th1_mem, th1_spike)
            th2_mem, th2_spike = mem_update(self.tfc2, th1_spike, th2_mem, th2_spike)
            th3_mem, th3_spike = mem_update(self.tfc3, th2_spike, th3_mem, th3_spike)
            th3_sumspike += th3_spike
            
        outputs_temporal = th3_sumspike / spatial_window

        # ==========================
        # Adaptive Fusion
        # ==========================
        outputs, fusion_weight = self.fusion(outputs_spatial, outputs_temporal)
        
        # Store fusion weight for analysis (optional)
        self.last_fusion_weight = fusion_weight.item() if torch.is_tensor(fusion_weight) else fusion_weight
        
        return outputs


class HybridLIFGNN_AdaptiveFusion_WithIntermediate(nn.Module):
    """
    Enhanced version that also fuses intermediate features
    """
    def __init__(self, num_classes=20, time_length=250, k=3, device="cuda:0", 
                 sparse=True, fusion_type='simple'):
        super(HybridLIFGNN_AdaptiveFusion_WithIntermediate, self).__init__()
        
        in_planes, out_planes = cfg_cnn[0]
        
        # Spatial and temporal graph convolutions
        self.conv1 = TAGConv(in_planes, out_planes, K=3)
        self.conv2 = TAGConv(in_planes, out_planes, K=3)

        # Spatial stream
        self.fc1 = nn.Linear(cfg_s[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], num_classes)

        # Temporal stream
        self.tfc1 = nn.Linear(time_length * cfg_cnn[-1][1], cfg_fc[0])
        self.tfc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.tfc3 = nn.Linear(cfg_fc[1], num_classes)

        self.num_classes = num_classes
        self.time_length = time_length
        self.sparse = sparse
        self.fusion_type = fusion_type
        
        self.Sgraph = TactileSpatialGraph(k)
        self.Tgraph = TactileTemporalGraph(self.time_length, self.sparse)
        self.device = device
        
        # Multiple fusion modules at different levels
        if fusion_type == 'simple':
            self.fusion_fc1 = SimpleFusion()
            self.fusion_fc2 = SimpleFusion()
            self.fusion_output = SimpleFusion()
        elif fusion_type == 'gated':
            self.fusion_fc1 = GatedFusion(cfg_fc[0])
            self.fusion_fc2 = GatedFusion(cfg_fc[1])
            self.fusion_output = GatedFusion(num_classes)
        else:
            # For attention, only fuse at output level
            self.fusion_output = AttentionFusion(num_classes)
            self.use_intermediate_fusion = False
        
        self.use_intermediate_fusion = (fusion_type in ['simple', 'gated'])
        print(f"Model initialized with {fusion_type} fusion (intermediate fusion: {self.use_intermediate_fusion})")

    def forward(self, input, training=True):
        data = input.to(self.device)
        sizes = data.size()
        time_window = sizes[-1]
        batch_size = sizes[0]
        
        # Initialize states
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_s[0], cfg_cnn[0][1], device=self.device)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, self.num_classes, device=self.device)

        t1_mem = t1_spike = torch.zeros(batch_size, time_window, cfg_cnn[0][1], device=self.device)
        th1_mem = th1_spike = th1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=self.device)
        th2_mem = th2_spike = th2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=self.device)
        th3_mem = th3_spike = th3_sumspike = torch.zeros(batch_size, self.num_classes, device=self.device)

        # Spatial stream
        inputs = data.split(1, dim=len(sizes) - 1)
        for step in range(time_window):
            x = inputs[step].squeeze().to(self.device)
            graph_data = self.Sgraph(x)
            x = graph_data.x.to(self.device)
            edge_idxs = graph_data.edge_index.to(self.device)

            c1_mem, c1_spike = mem_update_conv(self.conv1, x, edge_idxs, c1_mem, c1_spike, bn=self.s_bn)
            x = c1_spike.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            h3_mem, h3_spike = mem_update(self.fc3, h2_spike, h3_mem, h3_spike)
            h3_sumspike += h3_spike

        outputs_spatial_fc1 = h1_sumspike / time_window
        outputs_spatial_fc2 = h2_sumspike / time_window
        outputs_spatial = h3_sumspike / time_window

        # Temporal stream
        data_temporal_inputs = data
        temporal_inputs = data_temporal_inputs.split(1, dim=1)
        spatial_window = sizes[1]
        
        for temporal_step in range(spatial_window):
            temporal_x = temporal_inputs[temporal_step].squeeze()
            temporal_x = temporal_x.transpose(1, 2).to(self.device)
            temporal_graph_data = self.Tgraph(temporal_x)
            temporal_x = temporal_graph_data.x.to(self.device)
            temporal_edge_idxs = temporal_graph_data.edge_index.to(self.device)
            
            t1_mem, t1_spike = mem_update_conv(self.conv2, temporal_x, temporal_edge_idxs, t1_mem, t1_spike, bn=self.t_bn)
            temporal_x = t1_spike.view(batch_size, -1)

            th1_mem, th1_spike = mem_update(self.tfc1, temporal_x, th1_mem, th1_spike)
            th1_sumspike += th1_spike
            th2_mem, th2_spike = mem_update(self.tfc2, th1_spike, th2_mem, th2_spike)
            th2_sumspike += th2_spike
            th3_mem, th3_spike = mem_update(self.tfc3, th2_spike, th3_mem, th3_spike)
            th3_sumspike += th3_spike
            
        outputs_temporal_fc1 = th1_sumspike / spatial_window
        outputs_temporal_fc2 = th2_sumspike / spatial_window
        outputs_temporal = th3_sumspike / spatial_window

        # Adaptive fusion at multiple levels
        if self.use_intermediate_fusion:
            fused_fc1, weight_fc1 = self.fusion_fc1(outputs_spatial_fc1, outputs_temporal_fc1)
            fused_fc2, weight_fc2 = self.fusion_fc2(outputs_spatial_fc2, outputs_temporal_fc2)
        
        outputs, fusion_weight = self.fusion_output(outputs_spatial, outputs_temporal)
        
        self.last_fusion_weight = fusion_weight.item() if torch.is_tensor(fusion_weight) else fusion_weight
        
        return outputs


if __name__ == '__main__':
    # Test the models
    print("Testing Adaptive Fusion Models...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_classes = 36
    time_length = 250
    
    # Create dummy input (batch_size, 39, 2, time_length)
    dummy_input = torch.randn(batch_size, 39, 2, time_length).to(device)
    
    print("\n=== Testing Simple Fusion ===")
    model_simple = HybridLIFGNN_AdaptiveFusion(
        num_classes=num_classes, 
        time_length=time_length, 
        fusion_type='simple'
    ).to(device)
    output = model_simple(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Fusion weight (alpha): {model_simple.last_fusion_weight:.4f}")
    
    print("\n=== Testing Gated Fusion ===")
    model_gated = HybridLIFGNN_AdaptiveFusion(
        num_classes=num_classes, 
        time_length=time_length, 
        fusion_type='gated'
    ).to(device)
    output = model_gated(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Fusion weight (alpha): {model_gated.last_fusion_weight:.4f}")
    
    print("\n=== Testing Attention Fusion ===")
    model_attention = HybridLIFGNN_AdaptiveFusion(
        num_classes=num_classes, 
        time_length=time_length, 
        fusion_type='attention'
    ).to(device)
    output = model_attention(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Fusion weight (alpha): {model_attention.last_fusion_weight:.4f}")
    
    print("\nAll models tested successfully!")

