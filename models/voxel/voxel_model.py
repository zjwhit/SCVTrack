import torch
import torch.nn as nn
import torch.nn.functional as F

from .voxel_util import VoxelSetAbstraction


class get_model(nn.Module):
    def __init__(self,in_channel, nsample, overlap, divide):
        super(get_model, self).__init__()
        self.new_xyz = self.cal_new_xyz(divide)
        self.sa1 = VoxelSetAbstraction(nsample, overlap/2, in_channel, [64, 128, 256], [256, 256], divide, self.new_xyz)

    def cal_new_xyz(self, divide):
        x_num, y_num, z_num = divide[0], divide[1], divide[2]
        new_xyz = torch.zeros((x_num*y_num*z_num, 3), dtype=torch.float32)
        j = 0
        
        corner_x =  -x_num/2 + 1/2 
        corner_y =  -y_num/2 + 1/2 
        corner_z =  -z_num/2 + 1/2

        for x in range(0, x_num):
            for y in range(0, y_num):
                for z in range(0, z_num):
                    tmp_x = corner_x + x
                    tmp_y = corner_y + y
                    tmp_z = corner_z + z
                    new_xyz[j] = torch.tensor([tmp_x, tmp_y, tmp_z])
                    j = j + 1
        return new_xyz


    def forward(self, xyz, search_area):
        new_xyz, new_points = self.sa1(xyz, search_area)

        return new_xyz, new_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


