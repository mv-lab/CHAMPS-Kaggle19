import sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy

#define the device globally for all operations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
type_dict = {'1JHC': 2, '1JHN': 1, '2JHC': 3, '2JHH': 5, '2JHN': 7, '3JHC': 4,'3JHH': 8, '3JHN': 6}

activation = nn.ReLU(True)

def norm_adj(adj):
    #takes the adjacency matrix and returns
    #the normalized version
    
    #first, sum over rows to get the
    #degree of each node
    degree = adj.sum(dim=-1)
    
    #now take the sqrt of the reciprocal
    #of each value in the degree matrix
    #padded rows are 0, we need to exclude them
    #to prevent nan values
    degree[degree > 0] = 1 / torch.sqrt(degree[degree > 0])
    
    #then multiply the degree norm by identity
    degree_norm = degree.unsqueeze(-1) * torch.eye(degree.size(-1)).float().cuda()
    
    #lastly, do the multiplications DAD
    norm_adj = torch.matmul(degree_norm, adj)
    norm_adj = torch.matmul(norm_adj, degree_norm)
    
    return norm_adj

class EdgeConv(nn.Module):
    def __init__(self, nin, nout):
        super(EdgeConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1, bias=False)
        
    def forward(self, adj, dist):
        #apply the convolution to our dist matrix
        dist = self.conv(dist)
        
        #next multiply the new dist features
        #with the adj matrix
        #this defines individual neighborhoods
        out = dist * adj
        
        #unfortunately, we need to 
        #eliminate the zero padding
        #or the softmax function will be wrong,
        #the easiest way is to make all
        #the zero values in the adj matrix
        #very large negative numbers
        mask = (adj == 0).expand(out.size()).float()
        out = (-1e10 * mask) + out
        
        #lastly, apply the softmax function
        #to weight the output in 0-1, over each row
        #large negative exponentials go to zero
        out = F.softmax(out, dim=-1)
        
        #but to guarantee that padding is zeroed
        #let's use the adj matrix one last time
        mask = adj.expand(out.size())
        out = out * mask
        
        return out
    
class EdgeConvAttn(nn.Module):
    def __init__(self, nin, nout):
        super(EdgeConvAttn, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2, True)
        
    def forward(self, adj, node_mat, dist):
        #the output from a node convolution
        #is a (B, C, W, W) matrix, we take the transpose
        #of this matrix and concatenate it along the
        #channel dimension this gives us a matrix
        #in which each element is a pairwise relationship
        #as an added layer, let's also concatenate the edge
        #information like distance and coupling type
        node_pairs = torch.cat([node_mat, node_mat.transpose(-1, -2).contiguous(), dist], dim=1)
        
        #now we apply the attention matrix (aka, our 1x1 convolution)
        #followed by the leakyrelu
        node_pairs = self.conv(node_pairs)
        node_pairs = self.leaky(node_pairs)
        
        #next multiply node_pairs matrix
        #with the adj matrix
        #this defines individual neighborhoods
        out = node_pairs * adj
        
        #unfortunately, we need to 
        #eliminate the zero padding
        #or the softmax function will be wrong,
        #the easiest way is to make all
        #the zero values in the adj matrix
        #very large negative numbers
        mask = (adj == 0).expand(out.size()).float()
        out = (-9e15 * mask) + out
        
        #lastly, apply the softmax function
        #to weight the output in 0-1, over each row
        #large negative exponentials go to zero
        out = F.softmax(out, dim=-1)
        
        #but to guarantee that padding is zeroed
        #let's use the adj matrix one last time
        mask = adj.expand(out.size())
        out = out * mask
        
        return out
        
    
class NodeConv(nn.Module):
    def __init__(self, nin, nout):
        super(NodeConv, self).__init__()
        self.nin = nin
        self.convi = nn.Conv2d(nin, nout, 1, bias=False)
        self.convj = nn.Conv2d(nin, nout, 1, bias=False)
        
    def forward(self, adj, node):
        #convert node vector to a wxw matrix
        #where each row contains the set of all nodes
        w = adj.size(-1)
        adj = adj.repeat(1, self.nin, 1, 1)
        node = node.unsqueeze(-2).repeat(1, 1, w, 1)
        node = node * adj
        
        I = torch.eye(node.size(-1)).float().cuda()
        
        #apply the convolutions to our node features
        #convi gets applied to the node itself
        #convj gets applied to neighbors
        nodei = I * self.convi(node)
        nodej = (1 - I) * self.convj(node)
        node = nodei + nodej
        
        #values will be zeroed later when multiplied by
        #the edge weights
        return node
    
class NodeConvAttn(nn.Module):
    def __init__(self, nin, nout):
        super(NodeConvAttn, self).__init__()
        self.nin = nin
        self.conv = nn.Conv2d(nin, nout, 1, bias=False)
        
    def forward(self, adj, node):
        #convert node vector to a wxw matrix
        #where each row contains the set of all nodes
        w = adj.size(-1)
        adj = adj.repeat(1, self.nin, 1, 1)
        node = node.unsqueeze(-2).repeat(1, 1, w, 1)
        node = node * adj
        
        return self.conv(node)
    
class GraphConv(nn.Module):
    def __init__(self, edge_nin, node_nin, nout, dilation=1):
        super(GraphConv, self).__init__()
        self.dilation = dilation
        self.edge_conv = EdgeConv(edge_nin, nout)
        self.node_conv = NodeConv(node_nin, nout)

        
    def forward(self, adj, edge, node):
        #set the adj matrix to match the dilation rate
        padding_mask = (adj != 0).float().unsqueeze(1)
        adj = (adj == self.dilation)
        #make sure to include all diagonals as well
        I = torch.eye(adj.size(-1)).cuda().byte()

        adj = adj * (1 - I) + I
        adj = adj.float().unsqueeze(1) * padding_mask
        
        #first get the edge_weights
        edge_weights = self.edge_conv(adj, edge)
        
        #now get the node_features
        node_features = self.node_conv(adj, node)
        
        #multiply weights and node features
        #elementwise and sum over each node and its neighbors
        #just like a regular 2d convolution
        out = node_features * edge_weights
        out = out.sum(-1)
        return out
    
class GraphConvAttn(nn.Module):
    def __init__(self, edge_nin, node_nin, nout, dilation=1):
        super(GraphConvAttn, self).__init__()
        self.dilation = dilation
        self.edge_conv = EdgeConvAttn(edge_nin + nout * 2, nout)
        self.node_conv = NodeConvAttn(node_nin, nout)

        
    def forward(self, adj, edge, node):
        #set the adj matrix to match the dilation rate
        adj = (adj == self.dilation).float().unsqueeze(1)
        
        #first calculate the new node_matrix
        node_mat = self.node_conv(adj, node)
        
        #now get the attn weights
        attn = self.edge_conv(adj, node_mat, edge)
        
        #multiply weights and node features
        #elementwise and sum over each node and its neighbors
        #just like a regular 2d convolution
        out = node_mat * attn
        out = out.sum(-1)
        return out
    
class Affinity1d(nn.Module):
    def __init__(self, nin, edge_nin, compress=2, bias=False):
        super(Affinity1d, self).__init__()

        #now define the two functions that make up f
        self.theta = nn.Conv1d(nin, nin // compress, 1)
        self.phi = nn.Conv1d(nin, nin // compress, 1)

        #get the final value of f as an nxn scalar matrix
        self.concat_project = nn.Conv2d(2 * (nin // compress) + edge_nin, 1, 1, bias=bias)
        self.n_channels = nin // compress
        
    def forward(self, adj, edges, x):
        batch_size = x.size(0)
        h = adj.size(-1)
        
        # (b, c, H, 1)
        theta_x = self.theta(x).view(batch_size, self.n_channels, -1, 1)
        # (b, c, 1, H)
        phi_x = self.phi(x).view(batch_size, self.n_channels, 1, -1)
        
        #expand the sizes of theta_x and phi_x such that they are the same
        theta_x = theta_x.repeat(1, 1, 1, h)
        phi_x = phi_x.repeat(1, 1, h, 1)
        
        #concatenate along the channel dimension
        #the concatenated features contain information
        #about each atom and its bonded neighbors
        #the edges matrix contains pairwise features
        #and coupling types (edges is something of a misnomer)
        concat_feature = torch.cat([edges, theta_x, phi_x], dim=1)
        #get f as a scalar matrix
        #(Bx1xHxH)
        f = self.concat_project(concat_feature)
        
        return f

class NonLocal1d(nn.Module):
    def __init__(self, nin, edge_nin, compress=2):
        super(NonLocal1d, self).__init__()
        
        self.g = nn.Conv1d(nin, nin // compress, 1)
        
        self.W = nn.Sequential(
            nn.Conv1d(nin // compress, nin, 1, bias=False),
            nn.BatchNorm1d(nin)
        )
        
        #initialize the weights and bias to 0, as specified
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)
        
        #now define the two functions that make up f
        #using the affinity layer
        self.affinity = Affinity1d(nin, edge_nin, compress, bias=False)
        
        self.n_channels = nin // compress
        self.act = activation
        
    def forward(self, adj, edges, x, padding_mask, node_mask):
        batch_size = x.size(0)
        #print(x[0, 0].detach().cpu())
        
        #first get the g
        g_x = self.g(x).view(batch_size, self.n_channels, -1)
        #need g in the format BxHxC
        g_x = g_x.permute(0, 2, 1).contiguous() * node_mask
        
        f = self.act(self.affinity(adj, edges, x))
        #now we need to zero out padding elements only
        f = f * padding_mask
        f = f.squeeze()
    
        #this is the C(x) paramater in the paper
        N = node_mask.sum(dim=1).unsqueeze(1).repeat(1, f.size(-1), f.size(-1))
        f_div_C = f / N
        
        #(B, H, H)x(B, H, C) -> (B, H, C)
        y = torch.matmul(f_div_C, g_x)
        #print(f_div_C[0, 18].detach().cpu())
        
        #move the channel dimension back to the first position
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.n_channels, -1)
        W_y = self.W(y)
        
        return W_y + x
    
class gconv_bn_relu(nn.Module):
    def __init__(self, edge_nin, node_nin, nout, dilation=1):
        super(gconv_bn_relu, self).__init__()
        self.gc = GraphConv(edge_nin, node_nin, nout, dilation=dilation)
        self.bn = nn.BatchNorm1d(nout)
        self.act = activation
        
    def forward(self, adj, edge, node):
        x = self.gc(adj, edge, node)
        
        return self.act(self.bn(x))
    
class gconv_bn_relu_attn(nn.Module):
    def __init__(self, edge_nin, node_nin, nout, dilation=1):
        super(gconv_bn_relu_attn, self).__init__()
        self.gc = GraphConvAttn(edge_nin, node_nin, nout, dilation=dilation)
        self.bn = nn.BatchNorm1d(nout)
        self.act = activation
        
    def forward(self, adj, edge, node):
        x = self.gc(adj, edge, node)
        
        return self.act(self.bn(x))
    
class GraphConvBlock(nn.Module):
    def __init__(self, edge_nin, node_nin, nout, compress=2):
        super(GraphConvBlock, self).__init__()
        self.gc1 = gconv_bn_relu(edge_nin, node_nin, nout)
        self.gc2 = gconv_bn_relu(edge_nin, nout, nout)
        self.non_local = NonLocal1d(nout, edge_nin, compress=compress)
        
    def forward(self, adj, edge, node):
        #first apply the graph convs
        x = self.gc1(adj, edge, node)
        x = self.gc2(adj, edge, x)
        
        return x
    
class GraphBottleneck(nn.Module):
    def __init__(self, edge_nin, node_nin, nout, compress=2):
        super(GraphBottleneck, self).__init__()
        self.gc = GraphConvBlock(edge_nin, node_nin, nout, compress=compress)
        self.non_local = NonLocal1d(nout, edge_nin, compress=compress)
        
    def forward(self, adj, edge, node, padding_mask):
        identity = node
        x = self.gc(adj, edge, node)
        
        return self.non_local(adj, edge, x + identity, padding_mask)
    
class GraphBottleneckDilated(nn.Module):
    def __init__(self, edge_nin, node_nin,  node_out, dilation=1, compress=2, downsample=None):
        super(GraphBottleneckDilated, self).__init__()
        self.act = activation
        
        self.n_inter = node_nin // compress
        #first dimension reduction
        self.project_down = nn.Conv1d(node_nin, self.n_inter, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.n_inter)
        
        #second apply the graph convolution and non local layer
        self.gc = gconv_bn_relu(edge_nin, self.n_inter, self.n_inter, dilation)
        self.non_local = NonLocal1d(self.n_inter, edge_nin, compress=compress)
        
        #now project the dimensions back to the original number
        self.project_up = nn.Conv1d(self.n_inter, node_out, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(node_out)
        self.downsample = downsample
        
    def forward(self, x):
        adj = x['adj']
        edge = x['edges']
        node = x['nodes']
        padding_mask = x['padding_mask']
        node_mask = x['node_mask']
        
        identity = node
        
        #project down
        node = self.project_down(node)
        node = self.act(self.bn1(node))
        
        #apply graph convolution
        node = self.gc(adj, edge, node)
        node = self.non_local(adj, edge, node, padding_mask, node_mask)
        
        #project back up
        node = self.project_up(node)
        node = self.bn2(node)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x['nodes'] = self.act(node + identity)
        return x
    
class PairwiseConv(nn.Module):
    def __init__(self, n_types, edge_nin, node_nin):
        super(PairwiseConv, self).__init__()
        
        self.affinity = nn.Sequential(
            nn.Conv2d(edge_nin + node_nin * 2, node_nin // 2, 1, bias=False),
            nn.BatchNorm2d(node_nin // 2),
            activation
        )
        
        self.out_layer = nn.Conv2d(node_nin // 2, n_types, 1, bias=True)
        
    def forward(self, edge, node):
        #first we take our nodes in the shape
        #(B, C, W) and run them through conv1 and conv2
        #and them expand them in opposite directions
        w = node.size(-1)
        #f = self.f_x(node)
        #g = self.g_x(node)
        f = node.unsqueeze(-2).repeat(1, 1, w, 1)
        g = node.unsqueeze(-1).repeat(1, 1, 1, w)
            
        #now concatenate f and g so that we have an affinity matrix
        pairs = torch.cat([f, g, edge], dim=1)
        
        #now apply the affinity to get an asymmetric 
        #feature matrix of size (B, C, H, H)
        asym = self.affinity(pairs)
        
        #taking the asym matrix and adding it to its tranpose will give us
        #a symmetric output matrix
        sym = asym + asym.transpose(-1, -2)
        
        return self.out_layer(sym)
    
class MoleculePooling(nn.Module):
    def __init__(self, nin, nout):
        super(MoleculePooling, self).__init__()
        
        self.pool = nn.Sequential(
            nn.Conv1d(nin, nout, 1, bias=False),
            nn.BatchNorm1d(nout),
            activation
        )
        
    def forward(self, node_mask, node):
        h = node.size(-1)
        
        #get a count of the number of non-zero nodes
        #in the molecule (B, 1, H) --> (B, 1)
        n_atoms = node_mask.sum(1)
        
        #take the sum over all nodes and average
        #node: (B, C, H) --> (B, C)
        pooled = node.sum(-1) / n_atoms
        
        #lastly expand the size to (B, C, 1)
        pooled = pooled.unsqueeze(-1)
        
        #apply the convolution operations
        return self.pool(pooled)
    
class CouplingHead(nn.Module):
    def __init__(self, edge_nin, node_nin, nout):
        super(CouplingHead, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(edge_nin + node_nin, node_nin // 2, 1, bias=False),
            nn.BatchNorm2d(node_nin // 2),
            activation,
            nn.Conv2d(node_nin // 2, nout, 1, bias=True)
        )
        
    def forward(self, x):
        return self.layer(x)


class SemGCN(nn.Module):
    def __init__(self, n_types, atom_embedding_size, n_bond_embedding_size, inplanes=128, 
                 n_blocks=12, dilations=[1, 2, 3]):
        super(SemGCN, self).__init__()
        
        self.n_types = n_types
        #let's define two embedding layers, 1 for our bonds and the other for coupling types
        #5 types of atoms, 8 types of couplings, always include +1 to account for zeros
        self.atom_embed = nn.Embedding(17 + 1, atom_embedding_size, padding_idx=0)
        self.n_bond_embed = nn.Embedding(11 + 1, n_bond_embedding_size, padding_idx=0)
        self.type_embed = nn.Embedding(8 + 1, 5, padding_idx=0)
        
        edge_nin = 3 + n_bond_embedding_size + 5 #1 for distance, 2 for angles = 3
        node_nin = atom_embedding_size
        
        self.conv_in = gconv_bn_relu(edge_nin, node_nin, inplanes, dilation=1)
        self.non_local = NonLocal1d(inplanes, edge_nin, compress=2)
        
        #now define our network layers
        self.layer1 = self._make_layers(edge_nin, inplanes, inplanes * 2, n_blocks // 4, dilations)
        self.layer2 = self._make_layers(edge_nin, inplanes * 2, inplanes * 4, n_blocks // 4, dilations)
        self.layer3 = self._make_layers(edge_nin, inplanes * 4, inplanes * 8, n_blocks // 4, dilations)
        self.layer4 = self._make_layers(edge_nin, inplanes * 8, inplanes * 16, n_blocks // 4, dilations)
        self.molpool = MoleculePooling(inplanes * 16, inplanes)
        
        self.conv_out = CouplingHead(edge_nin, inplanes * 16 + inplanes, n_types)
        
    def _make_layers(self, edge_nin, inplanes, outplanes, n_blocks, dilations):
        downsample = nn.Sequential(
                nn.Conv1d(inplanes, outplanes, 1, bias=False),
                nn.BatchNorm1d(outplanes)
        )
        layers = [GraphBottleneckDilated(edge_nin, inplanes, outplanes, dilation=1, downsample=downsample)]
        for ix in range(1, n_blocks):
            dilation = dilations[ix % len(dilations)]
            layers.append(GraphBottleneckDilated(edge_nin, outplanes, outplanes, dilation=dilation))
        
        return nn.Sequential(*layers)
        
    def forward(self, distances, bonds, types, angles, adj):
        #embeddings puts the channel dimension last
        bsz = bonds.size(-1)
        h = bonds.size(1)
        assert (len(distances.size()) == 4), 'Unsqueeze channel dim of distances!'
        assert (len(adj.size()) == 3), 'Remove channel dim from graph!'
        
        #first get a padding mask
        padding_mask = (adj != 0).float().unsqueeze(1)
        node_mask = (bonds != 0).float().unsqueeze(1).permute(0, 2, 1).contiguous()
        
        atoms = self.atom_embed(bonds).transpose(-1, 1)
        types = self.type_embed(types).transpose(-1, 1)
        n_bonds = self.n_bond_embed(adj).transpose(-1, 1)
        
        edges = torch.cat([distances, angles, n_bonds, types], dim=1)
        nodes = atoms
        
        x = self.conv_in(adj, edges, nodes)
        x = self.non_local(adj, edges, x, padding_mask, node_mask)
        feature_dict = {'adj': adj, 'edges': edges, 'nodes': x, 'padding_mask': padding_mask, 'node_mask': node_mask}
        
        #x = self.layers(feature_dict)
        x = self.layer1(feature_dict)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        nodes = x['nodes']
        
        #get our node features into hxh matrix
        #print(nodes[0, 0].detach().cpu())
        #pooled: (B, C, 1) --> (B, C, H, H)
        pooled = self.molpool(node_mask, nodes).unsqueeze(-1).repeat(1, 1, h, h)
        out = torch.matmul(nodes.unsqueeze(-1), nodes.unsqueeze(-2))
        out = torch.cat([edges, out, pooled], dim=1)
        out = self.conv_out(out)
        return out
    
class SmoothL1(nn.Module):
    def __init__(self, beta=0.1):
        super(SmoothL1, self).__init__()
        """At later time check if we should handle error by image or batch"""
        self.beta = beta

    def forward(self, output, target, types, size_average=True):
        #first we need to filter our all the zeros from padding and 
        #empty scalar constants
        bsz = output.size(0)
        n_types = output.size(1)

        #one hot encode the types
        h = types.size(-1)
        k = torch.arange(1, n_types + 1).view(n_types, 1, 1).repeat(1, h, h).cuda().long()
        type_mask = (k == types.unsqueeze(1)).type(output.dtype)

        #now multiply the output and target by the type mask
        output = type_mask * output
        target = type_mask * target

        n = torch.abs(output - target)
        cond = n < self.beta
        mae_type = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        #mae_type = mae_type.view(bsz, n_types, -1).permute(1, 0, 2).contiguous()

        #now take the mean of each channel and sum over all channels
        #mae_type_sum = torch.sum(mae_type.view(n_types, -1), dim=-1)
        #mae_type_count = torch.sum(mae_type.view(n_types, -1) != 0, dim=-1).type(mae_type_sum.dtype)

        mae_sum = mae_type.sum()
        mae_count = (mae_type != 0).sum()

        return torch.log(mae_sum / mae_count)

class SmoothL1Types(nn.Module):
    def __init__(self, beta=0.1):
        super(SmoothL1Types, self).__init__()
        """At later time check if we should handle error by image or batch"""
        self.beta = beta
        
    def forward(self, output, target, types, size_average=True):
        #first we need to filter our all the zeros from padding and 
        #empty scalar constants
        bsz = output.size(0)
        n_types = output.size(1)
        
        #one hot encode the types
        h = types.size(-1)
        k = torch.arange(1, n_types + 1).view(n_types, 1, 1).repeat(1, h, h).cuda().long()
        type_mask = (k == types.unsqueeze(1)).type(output.dtype)
        
        #now multiply the output and target by the type mask
        output = type_mask * output
        target = type_mask * target
        
        n = torch.abs(output - target)
        cond = n < self.beta
        mae_type = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        mae_type = mae_type.view(bsz, n_types, -1).permute(1, 0, 2).contiguous()
        
        #now take the mean of each channel and sum over all channels
        mae_type_sum = torch.sum(mae_type.view(n_types, -1), dim=-1)
        mae_type_count = torch.sum(mae_type.view(n_types, -1) != 0, dim=-1).type(mae_type_sum.dtype)
        
        mae_type = mae_type_sum / (mae_type_count + 1)
        
        mae_type = torch.mean(mae_type)
        
        return mae_type
    
class MSETypes(nn.Module):
    def __init__(self, reduction='sum'):
        super(MSETypes, self).__init__()
        """At later time check if we should handle error by image or batch"""
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        
    def forward(self, output, target, types):
        #first we need to filter our all the zeros from padding and 
        #empty scalar constants
        bsz = output.size(0)
        n_types = output.size(1)
        
        #one hot encode the types
        h = types.size(-1)
        k = torch.arange(1, n_types + 1).view(n_types, 1, 1).repeat(1, h, h).cuda().long()
        type_mask = (k == types.unsqueeze(1)).type(output.dtype)
        
        #now multiply the output and target by the type mask
        output = type_mask * output
        target = type_mask * target
        
        mse_type = self.mse(output, target)
        #print(output[target > 0][:5], target[target > 0][:5], mse_type[target > 0][:5])
        mse_type = mse_type.view(bsz, n_types, -1).permute(1, 0, 2).contiguous()
        
        #now take the mean of each channel and sum over all channels
        mse_type_sum = torch.sum(mse_type.view(n_types, -1), dim=-1)
        mse_type_count = torch.sum(mse_type.view(n_types, -1) != 0, dim=-1).type(mse_type_sum.dtype)
        
        mse_type = mse_type_sum / (mse_type_count + 1)
        
        mse_type = torch.mean(mse_type)
        
        return mse_type
    
class Learner:
    
    def __init__(self, model, loss, optimizer, train, valid=None):
        model = deepcopy(model)
        self.model = model
        self.loss = loss
        self.optim = optimizer
        self.optimizer = self.optim(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        self.train = train
        self.valid = valid
        
        self.train_iter = iter(train)
        if valid is not None:
            self.valid_iter = iter(valid)
        
        #only metric besides loss that we care about is MAE
        #we'll store the values in a list that get's refreshed
        #when metrics are printed
        self.train_metrics = {'loss': [], 'mae': []}
        self.valid_metrics = {'val_loss': [], 'val_mae': []}
        self.strikes = 0
        
    def load_batch(self, dataloader):
        return dataloader.next()
    
    def extract_data(self, batch):
        #4 fields that we care about in the batch: distance, bond, type, and target
        distances = batch['distance'].unsqueeze(1).to(device)
        angles = batch['angle'].to(device)
        atoms = batch['atoms'].to(device)
        types = batch['type'].to(device)
        graph = batch['graph'].to(device)
        targets = batch['target'].unsqueeze(1).to(device)
        
        #distances and targets need to have a channel dimension added, bonds and types
        #have channel dimension added when then pass through the embedding layer
        
        return [distances, atoms, types, angles, graph], targets
    
    def batch_train(self):
        #first get a batch of data from training set
        try:
            batch = self.load_batch(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train)
            batch = self.load_batch(self.train_iter)
            
        in_data, targets = self.extract_data(batch)
        types = in_data[2]
        
        #now predict and update the model
        self.optimizer.zero_grad()
        output = self.model.train()(*in_data)
        l = self.loss(output, targets, types)
        l.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        
        self.optimizer.step()
        
        #lastly calculate the mae
        mae = calculate_mae(output.detach(), targets.detach(), types.detach())
        return l.item(), mae
    
    def batch_eval(self):
        #first get a batch of data from training set
        with torch.no_grad():
            try:
                batch = self.load_batch(self.valid_iter)
            except StopIteration:
                self.valid_iter = iter(self.valid)
                batch = self.load_batch(self.valid_iter)
                
            in_data, targets = self.extract_data(batch)
            types = in_data[2]

            #now predict and update the model
            output = self.model.eval()(*in_data)
            l = self.loss(output, targets, types)

            #lastly calculate the mae
            mae_sum, mae_count = calculate_mae(output.detach(), targets.detach(), types.detach(), return_counts=True)
            
        return l.item(), mae_sum, mae_count
            
    def batch_predict(self, batch):
        with torch.no_grad():
            in_data, targets = self.extract_data(batch)

            #now predict and update the model
            output = self.model.eval()(*in_data)
            
        return output.detach().cpu(), targets.detach().cpu()
    
    def print_metrics(self, metric_dict):
        for m in metric_dict.keys():
            print('{}: {}'.format(m, metric_dict[m][-1]))
    
    def training(self, epochs, iters=None, lr=1e-3, save_path=None):
        #if no fixed number of iterations is specified we just use the length
        #of the training dataset
        if not iters:
            iters = len(self.train)
            
        self.optimizer.param_groups[0]['lr'] = lr
        
        rl = rmae = 0
        avg_mom = 0.98
        for e in range(epochs):
            rl = rmae = 0
            print('Epoch {}'.format(e + 1))
            for ix in tqdm(range(iters), file=sys.stdout):
                loss, mae = self.batch_train()
                rl = rl * avg_mom + loss * (1 - avg_mom)
                rmae = rmae * avg_mom + mae * (1 - avg_mom)
                debias_loss = rl / (1 - avg_mom ** (ix + 1))
                debias_mae = rmae / (1 - avg_mom ** (ix + 1))
                
            self.train_metrics['loss'].append(debias_loss)
            self.train_metrics['mae'].append(debias_mae)
            
            self.print_metrics(self.train_metrics)
            
            if self.valid:
                self.evaluate()
                
                #self.reduce_lr_plateau(self.valid_metrics['val_loss'])
                
                if save_path is not None:
                    if self.valid_metrics['val_loss'][-1] <= min(self.valid_metrics['val_loss']):
                        self.save_state(save_path)
                        print('Model saved to {}'.format(save_path))
    
    def train_one_cycle(self, total_iters=10000, max_lr=1e-3, wd=0.0, eval_iters=None, save_path=None):
        
        #if no eval iters are given, use 1 epoch as evaluation period
        if eval_iters is None:
            eval_iters = len(self.train)
        
        #first instantiate the LRCalculator
        schedule = OneCycle(total_iters, max_lr)
        lr, mom = schedule.calc()
        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[0]['betas'] = (mom, 0.99)
        self.optimizer.param_groups[0]['weight_decay'] = wd
        
        rl = rmae = 0
        avg_mom = 0.98
        for ix in tqdm(range(total_iters), file=sys.stdout):
            loss, mae = self.batch_train()
            lr, mom = schedule.calc()
            self.optimizer.param_groups[0]['lr'] = lr
            self.optimizer.param_groups[0]['betas'] = (mom, 0.99)
                
            rl = rl * avg_mom + loss * (1 - avg_mom)
            rmae = rmae * avg_mom + mae * (1 - avg_mom)
            debias_loss = rl / (1 - avg_mom ** (ix + 1))
            debias_mae = rmae / (1 - avg_mom ** (ix + 1))
            
            if (ix > 0) & ((ix + 1) % eval_iters == 0):
                try:
                    self.train_metrics['loss'].append(debias_loss)
                    self.train_metrics['mae'].append(debias_mae)
                except:
                    pass

                if save_path is not None:
                    #if self.train_metrics['loss'][-1] <= min(self.train_metrics['loss'])
                    torch.save(self.model.state_dict(), save_path)

                self.print_metrics(self.train_metrics)
                rl = rmae = 0
                
                if save_path is not None:
                    #if self.train_metrics['loss'][-1] <= min(self.train_metrics['loss'])
                    self.save_state(save_path)
                    print('Model saved to {}'.format(save_path))

                if self.valid:
                    self.evaluate()   
        
    
    def evaluate(self):
        assert (self.valid is not None), "No validation data to evaluate!"
        
        rl = 0
        rmae_sum = torch.zeros((self.model.n_types,), dtype=torch.float32)
        rmae_count = torch.zeros((self.model.n_types,), dtype=torch.float32)
        for ix in range(len(self.valid)):
            val_loss, val_mae_sum, val_mae_count = self.batch_eval()
            rl += val_loss
            rmae_sum += val_mae_sum.cpu()
            rmae_count += val_mae_count.cpu()
            
        mae_type = torch.log(rmae_sum / rmae_count)
        print(mae_type)
        rmae = torch.mean(mae_type)

        self.valid_metrics['val_loss'].append(rl / len(self.valid))
        self.valid_metrics['val_mae'].append(rmae)
        self.print_metrics(self.valid_metrics)
        
    def create_submission(self, test_data, test_df):
        group = test_df.groupby(by='molecule_name')
        mnames = np.array(list(group.groups.keys()))
        indices = np.array(list(group.groups.values()))
        ids = test_df['id'].values
        atom_0s = test_df['atom_index_0'].values
        atom_1s = test_df['atom_index_1'].values

        out_ids = []
        out_preds = []
        for batch in tqdm(test_data):
            molecule = batch['fname'][0].split('.')[0]
            pred_map, _ = self.batch_predict(batch)
            pred_map = pred_map.squeeze().numpy()
            
            #now find molecule name in our test_df
            mnames_index = np.where(mnames == molecule)[0]
            mol_indices = indices[mnames_index][0].tolist()
            x, y = atom_0s[mol_indices], atom_1s[mol_indices]
            channels = batch['type'].squeeze().numpy()[(x, y)] - 1
            
            #now get the predictions required
            predictions = pred_map[(channels, atom_0s[mol_indices], atom_1s[mol_indices])]
            
            #append predictions and ids
            out_ids.extend(ids[mol_indices].tolist())
            out_preds.extend(predictions.tolist())
        
        pred_df = pd.DataFrame(data=out_preds, columns=['scalar_coupling_constant'])
        pred_df.index = out_ids
        pred_df.index.name = 'id'
        return pred_df.sort_values(by='id')
        
    def range_test(
        self,
        start_lr,
        end_lr=10,
        wd=0.0,
        num_iter=100,
        smooth_f=0.05,
        diverge_th=5,
    ):
        """Performs the learning rate range test.

        Arguments:
            train_loader (torch.utils.data.DataLoader): the training set data laoder.
            val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
                will only use the training loss. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation loss
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.

        """
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.optimizer.param_groups[0]['lr'] = start_lr
        self.optimizer.param_groups[0]['weight_decay'] = wd

        # Initialize the proper learning rate policy
        lr_schedule = ExponentialLR(start_lr, end_lr, num_iter)

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        # Create an iterator to get data batch by batch
        for iteration in tqdm(range(num_iter)):
            # Train on batch and retrieve loss
            loss, mae = self.batch_train()

            # Update the learning rate
            new_lr = lr_schedule.get_lr()
            self.optimizer.param_groups[0]['lr'] = new_lr
            self.history["lr"].append(new_lr)

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

        self.plot()
        
    def plot(self, skip_start=1, skip_end=1, log_lr=True):
        """Plots the learning rate range test.

        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.

        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()
            
    def save_state(self, save_path):
        state = {'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, save_path)
        
    def load_state(self, load_path):
        state = torch.load(load_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        
    def to_cuda(self):
        self.model = self.model.to(device)
        
        
def calculate_mae(output, target, types, return_counts=False):
    #first, let's flatten our input
    bsz = output.size(0)
    n_types = output.size(1)

    #one hot encode the types
    h = types.size(-1)
    k = torch.arange(1, n_types + 1).view(n_types, 1, 1).repeat(1, h, h).cuda().long()
    type_mask = (k == types.unsqueeze(1)).type(output.dtype)

    #now multiply the output and target by the type mask
    output = type_mask * output
    target = type_mask * target
    mae_type = torch.abs(output - target).view(bsz, n_types, -1).permute(1, 0, 2).contiguous()
    
    mse_type = mae_type.view(bsz, n_types, -1).permute(1, 0, 2).contiguous()
        
    #now take the mean of each channel and sum over all channels
    mae_type_sum = torch.sum(mae_type.view(n_types, -1), dim=-1)
    mae_type_count = torch.sum(mae_type.view(n_types, -1) != 0, dim=-1).type(mae_type_sum.dtype)

    if return_counts:
        return mae_type_sum, mae_type_count + 1
    
    mae_type = (mae_type_sum + 1e-3) / (mae_type_count + 1e-3)
    mae_type = torch.log(mae_type)
    
    return mae_type.mean()


class LinearLR:
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, start_lr, end_lr, num_iter):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.step_size = (end_lr - start_lr) / num_iter
        self.iteration = 0

    def get_lr(self):
        self.iteration += 1
        return self.start_lr + (self.step_size * self.iteration)
    
class ExponentialLR:
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, start_lr, end_lr, num_iter):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.alpha = (1 / num_iter) * np.log(end_lr / start_lr)
        self.iteration = 0

    def get_lr(self):
        self.iteration += 1
        return self.start_lr * np.exp(self.alpha * self.iteration)
    
class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during 
    whole run with 2 steps of equal length. During first step, increase the learning rate 
    from lower learning rate to higher learning rate. And in second step, decrease it from 
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one 
    addition to this. - During last few hundred/thousand iterations of cycle reduce the 
    learning rate to 1/100th or 1/1000th of the lower learning rate.

    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make 
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is 
    increasing and increase momentum when learning rate is decreasing.

    Args:
        nb              Total number of iterations including all epochs

        max_lr          The optimum learning rate. This learning rate will be used as highest 
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.

        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)

        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10

        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning 
                        rate below lower learning rate.
                        The default value is 10.
    """
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
        self.nb = nb
        self.div = div
        self.step_len =  int(self.nb * (1- prcnt/100)/2)
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []
        
    def calc(self):
        self.iteration += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return (lr, mom)
        
    def calc_lr(self):
        if self.iteration==self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        if self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            lr = self.high_lr * ( 1 - 0.99 * ratio)/self.div
        elif self.iteration > self.step_len:
            ratio = 1- (self.iteration -self.step_len)/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else :
            ratio = self.iteration/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr
    
    def calc_mom(self):
        if self.iteration==self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        if self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else :
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom

class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

        return loss
