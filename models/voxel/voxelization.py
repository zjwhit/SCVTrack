import torch
import torch.nn as nn
from .functional.voxelization import avg_voxelize,favg_voxelize

# __all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, divide, mode=True):
        super().__init__()
        self.divide = torch.Tensor(divide)
        self.x = divide[0]
        self.y = divide[1]
        self.z = divide[2]
        self.min_voxel_coord = -1 * torch.floor(self.divide/ 2)
        self.mode=mode

    def forward(self, features, coords, search_area):
        #(b,c,n)(b,n,3)
        # search_area.detach()
        device = coords.device
        search_area.to(device)
        
        voxel_size = search_area / torch.Tensor([20, 20, 20]).to(device)
        coords_detach = coords.detach()
        B, N, _ = coords.shape
        discrete_pts = torch.floor(coords_detach / voxel_size.reshape(B, 1, 3).repeat(1, N, 1))
        voxel_indices = (discrete_pts - self.min_voxel_coord.to(device)).int()
        voxel_indices[voxel_indices < 0] = 0
        voxel_indices[voxel_indices > 19] = 19
        voxel_indices=voxel_indices.transpose(1, 2).contiguous()
        if self.mode:
            return favg_voxelize(features, voxel_indices, self.x,self.y,self.z)
        else:
            return avg_voxelize(features, voxel_indices, self.x, self.y, self.z)

    def extra_repr(self):
        print('information:x {} y {} z {} min_voxel_coord {} voxel_size {} '.format(self.x,self.y,self.z,self.min_voxel_coord,self.voxel_size))