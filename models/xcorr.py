import torch
from torch import nn
import torch.nn.functional as F
from .torch_util import Conv2d,Conv3d
from collections import OrderedDict

class PN_XCorr(nn.Module):
    def __init__(self, input_channel, per_point_mlp, hidden_mlp, output_size=0):
        super(PN_XCorr, self).__init__()
        seq_per_point = []
        in_channel = input_channel
        for out_channel in per_point_mlp:
            seq_per_point.append(nn.Conv1d(in_channel, out_channel, 1))
            seq_per_point.append(nn.BatchNorm1d(out_channel))
            seq_per_point.append(nn.ReLU())
            in_channel = out_channel
        in_channel = in_channel * 20 * 20 * 1 
        seq_hidden = []
        for out_channel in hidden_mlp:
            seq_hidden.append(nn.Linear(in_channel, out_channel))
            seq_hidden.append(nn.BatchNorm1d(out_channel))
            seq_hidden.append(nn.ReLU())
            in_channel = out_channel
        
        self.features = nn.Sequential(*seq_per_point,
                                    #   nn.AdaptiveMaxPool1d(output_size=1),
                                      nn.Flatten(),
                                      *seq_hidden)
        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Linear(in_channel, output_size)

    def forward(self, prev_frame):
        """
        :param template_feature: B,f,M
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :return:
        """
        B = prev_frame.size(0)
        f = prev_frame.size(1)
        n1 = prev_frame.size(2)
        
        x = self.features(prev_frame)
        if self.output_size > 0:
            x = self.fc(x)

        return x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        # xyz : BxNx3
        # xyz = xyz.transpose(1, 2).contiguous()
        # Bx3xN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class Relationnet(nn.Module):
    def __init__(self, input_channel, hidden_mlp, output_size=0):
        super(Relationnet, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=1)
        in_channel = input_channel 
        seq_hidden = []
        for out_channel in hidden_mlp:
            seq_hidden.append(nn.Linear(in_channel, out_channel))
            seq_hidden.append(nn.BatchNorm1d(out_channel))
            seq_hidden.append(nn.ReLU())
            in_channel = out_channel
        
        self.features = nn.Sequential(*seq_hidden)
        self.output_size = output_size
        if output_size >= 0:
            self.fc = nn.Linear(in_channel, output_size)
        

    def forward(self, prev_frame, curr_frame):
        """

        :param template_feature: B,f,M ->B,256,x*y*z
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :return:
        """
        
        B = prev_frame.size(0)
        f = prev_frame.size(1)
        n1 = prev_frame.size(2)
        n2 = curr_frame.size(2)
        final_out_cla = self.cosine(prev_frame.unsqueeze(-1).expand(B, f, n1, n2),
                             curr_frame.unsqueeze(2).expand(B, f, n1, n2))  # B,n1,n2

        prev = torch.max(final_out_cla,2).values
        prev_mask = torch.argmax(torch.max(final_out_cla,2).values, 1, keepdim=True).unsqueeze(1).expand(B, f, 1) # B, 1
        curr_mask = torch.argmax(torch.max(final_out_cla,1).values, 1, keepdim=True).unsqueeze(1).expand(B, f, 1) # B, 1
        
        prev_feature = torch.gather(prev_frame, dim=2, index=prev_mask)
        curr_feature = torch.gather(curr_frame, dim=2, index=curr_mask)
        fusion = torch.cat((prev_feature, curr_feature), dim=1) # B, 2048, 1
        fusion = fusion.view(B, 2048)
        # fusion = final_out_cla
        x = self.features(fusion)
        if self.output_size > 0:
            x = self.fc(x)
        return x


class VoxelConv(nn.Module):
    def __init__(self, divide):
        super().__init__()
        self.divide = divide[0]
        self.conv2d_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1)
        self.with_pos_embed = False
        if self.with_pos_embed:
            self.encoder_pos_embed = PositionEmbeddingLearned(3, 256)
        else:
            self.encoder_pos_embed = None

    def forward(self, prev_frame, curr_frame, prev_xyz, curr_xyz):
        """

        :param template_feature: B,f,M ->B,256,x*y*z
        :param search_feature: B,f,N
        :param template_xyz: B,M,3
        :return:
        """
        
        B = prev_frame.size(0)
        f = prev_frame.size(1)
        if self.with_pos_embed:
            prev_frame = self.encoder_pos_embed(prev_xyz) + prev_frame
            curr_frame = self.encoder_pos_embed(curr_xyz) + curr_frame
        
        prev_frame = prev_frame.reshape(B, f, self.divide, self.divide)
        curr_frame = curr_frame.reshape(B, f, self.divide, self.divide)
        conv_prev_frame = self.conv2d_2(self.conv2d_1(prev_frame)) # 10 * 10 --> 8 * 8 --> 6 * 6
        conv_curr_frame = self.conv2d_2(self.conv2d_1(curr_frame))
        prev_frame = conv_prev_frame.reshape(B, 256, (self.divide-2*2) ** 2)
        curr_frame = conv_curr_frame.reshape(B, 256, (self.divide-2*2) ** 2)

        return prev_frame, curr_frame

class Conv_Middle_layers(nn.Module):
    def __init__(self,inplanes):
        super(Conv_Middle_layers, self).__init__()
        self.conv1 = Conv3d(inplanes, 64, stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv2 = Conv3d(64, 64, stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv3 = Conv3d(64, 64, stride=(2, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Sequential(OrderedDict([
            # ('conv3d',nn.Conv3d(64,128,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))),
            ('conv3d',nn.Conv3d(64,128,kernel_size=(3,1,1),stride=(1,1,1),padding=(0,0,0))),
            ('bn',nn.BatchNorm3d(128)),
            ('relu',nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out=self.conv4(out)
        shape = out.size()
        # print("conv3d feature before maxpool: {}".format(shape))
        out=F.max_pool3d(out,kernel_size=[shape[2], 1, 1])
        out=out.squeeze(2)
        # print("conv3d feature size: {}".format(out.size()))
        out = out.reshape(shape[0], shape[1], shape[3]*shape[4])
        return out