from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def in_voxel(new_xyz, xyz, wlh):
    """
    wlh: [B, 3]
    """
    B, S, C = new_xyz.shape
    _, N, _ = xyz.shape
    device = xyz.device
    w = wlh[:, 0]
    wlh[:, 0] = wlh[:, 1]
    wlh[:, 1] = w
    lwh = wlh
    new_xyz = new_xyz.view(B, S, 1, C).repeat(1, 1, N, 1)
    xyz = xyz.view(B, 1, N, C).repeat(1, S, 1, 1)
    difference = torch.sub(xyz, new_xyz)
    lwh = lwh.view(B, 1, C).repeat(1, S, 1)
    lwh = lwh.view(B, S, 1, C).repeat(1, 1, N, 1)
    max = torch.max(torch.sub(torch.abs(difference), lwh), dim=3) 
    in_or_not = max[0]
    
    return in_or_not


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(nsample, xyz, new_xyz, search_area, x_num, y_num, z_num, overlap):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    wlh = search_area.to(device)
    divide = torch.tensor([y_num/overlap , x_num/overlap , z_num/overlap]).to(device) 
    wlh = torch.div(wlh, divide)
    sqrdists = in_voxel(new_xyz, xyz, wlh)
    group_idx[sqrdists > 0] = 0
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]   
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == 0
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def get_new_xyz(search_areas, npoint, divide, nsample, xyz, overlap, pre_xyz):
    # search_areas [B, 3]: wlh
    x_num, y_num, z_num = divide[0], divide[1], divide[2]
    B, N, C = xyz.shape
    S = npoint
    device = xyz.device
    
    pre_xyz = pre_xyz.to(device)
    pre_xyz = pre_xyz.view(1,S,3).repeat(B,1,1)
    lwh = search_areas.to(device)
    lwh = torch.div(lwh, torch.Tensor(divide).to(device)).view(B,1,3).repeat(1,S,1)
    new_xyz = torch.mul(pre_xyz, lwh)

    xyz_coor = xyz[:, :, :3]
    xyz_feature = xyz[:, :, 3:]
    idx = query_ball_point(nsample, xyz_coor, new_xyz, search_areas, x_num, y_num, z_num, overlap)
    grouped_xyz = index_points(xyz_coor, idx) # [B, npoint, nsample, 3]
    grouped_xyz_feature = index_points(xyz_feature, idx)    # [B,npoint,nsamle,C-3]
    grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3)
    new_points = torch.cat((grouped_xyz, grouped_xyz_feature), dim=-1)  #B,npoint,nsamle,C-3 + 3->C

    return new_xyz, new_points


class VoxelSetAbstraction(nn.Module):
    def __init__(self, nsample, overlap, in_channel, mlp, hidden_mlp, divide, pre_xyz):
        super(VoxelSetAbstraction, self).__init__()
        self.divide = divide
        self.npoint = divide[0] * divide[1] * divide[2]
        self.nsample = nsample
        self.overlap = overlap
        self.pre_xyz = pre_xyz
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        seq_hidden = []
        for out_channel in hidden_mlp:
            seq_hidden.append(nn.Conv1d(last_channel, out_channel, 1))
            seq_hidden.append(nn.BatchNorm1d(out_channel))
            seq_hidden.append(nn.ReLU())
            last_channel = out_channel
        
        self.features = nn.Sequential(*seq_hidden)

    def forward(self, xyz, search_area):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # B,C,N->B,N,C

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        
        new_xyz, new_points = get_new_xyz(search_area, self.npoint, self.divide, self.nsample, xyz, self.overlap, self.pre_xyz)

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_points = self.features(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        # new_points = torch.cat([new_xyz, new_points], dim=1)  
        return new_xyz, new_points
        


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
        


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i] 
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)


        return new_xyz, new_points_concat

