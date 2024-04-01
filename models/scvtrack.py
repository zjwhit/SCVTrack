"""
m2track.py
Created by zenn at 2021/11/24 13:10
"""
import time
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet, Pointnet_Backbone
from models.xcorr import PN_XCorr, Relationnet, VoxelConv, Conv_Middle_layers
from torch import nn
from torchmetrics import Accuracy
from utils.metrics import estimateAccuracy, estimateOverlap

from .attention.pointnet_tranformer import PointnetTransformerSiamese
from .pointnetpp.pointnet2_cls_msg import get_model as PointNetPPCls
from .pointnetpp.pointnet2_sem_seg_msg import get_model as PointNetPP
from .voxel.voxel_model import get_model as Voxel
from .voxel.voxelization import Voxelization
from .voxel.voxel_mask import VoxelMaskEmbedding
from .kpconv.kpconv_voxel_model import get_model as KPConvVoxel
from .kpconv.KPConv import AnchorBackbone
from MemoryClassification.pointnet_cls import PointNet

class SCVTRACK(base_model.MotionBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.seg_acc = Accuracy(num_classes=2, average='none')
        self.category_name = getattr(config, 'category_name', 'Car')
        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_origin_second_stage = getattr(config, 'use_origin_second_stage', True)
        self.use_second_stage = self.use_origin_second_stage
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
        self.random_size = getattr(config, 'point_sample_size', 1024)
        self.divide = getattr(config, 'divide', [10, 10, 1])
        self.num_attention_layers = getattr(config, 'num_attention_layers', 2)
        self.use_memory_refine = getattr(config, 'use_memory_refine', False)
        self.use_memory_refine_after = getattr(config, 'use_memory_refine_after', False)
        self.use_memory_cls = getattr(config, 'use_memory_cls', False)

        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))

        # self.origin_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
        #                                   per_point_mlp=[64, 128, 256, 512],
        #                                   hidden_mlp=[512, 256],
        #                                   output_size=-1)

        self.backbone_net = Pointnet_Backbone(use_fps=True, input_channels=0)
        self.voxelization = Voxelization(divide = self.divide)
        self.conv_middle_layers = Conv_Middle_layers(inplanes=64)
        # self.voxel_mask = VoxelMaskEmbedding(divide = self.divide, num_pos_feats=128)

        self.attention = PointnetTransformerSiamese(num_layers = self.num_attention_layers, channel=128)
        self.xcorr = PN_XCorr(input_channel=128, 
                            #    per_point_mlp=[128, 256, 512],
                            per_point_mlp=[256, 128, 32, 4],
                            hidden_mlp=[512, 256],
                            output_size=-1)
        
        if self.use_memory_cls:
            self.memory_cls = PointNet(input_channel=3, per_point_mlp=[64, 128, 256, 512], hidden_mlp=[512, 256], output_size=1)

        if self.use_memory_refine:
            self.mini_pointnet = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                              per_point_mlp=[64, 128, 256, 512],
                                              hidden_mlp=[512, 256],
                                              output_size=-1)
            self.pre_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))

        if self.use_origin_second_stage:
            # the pointnet that source code use
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))

        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)
            "search_area": (B,3)
        }

        Returns: B,4

        """
        starttime = time.time()
        output_dict = {}
        x = input_dict["points"].transpose(1, 2) # 

        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2)   # B,9,N
            x = torch.cat([x, candidate_bc], dim=1) # B,9+5,N

        B, _, N = x.shape

        seg_out, seg_feature = self.seg_pointnet(x)  # B,4,N
        segtime = time.time() - starttime
        
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N

        mask_points = x[:, :3, :] * pred_cls
        # mask_xyz_t0 = mask_points[:, :, :N // 2]  # B,3,N//2
        # mask_xyz_t1 = mask_points[:, :, N // 2:]
        # s = int(N / 4 * 3)
        # mask_xyz_t0 = mask_points[:, :3, :s]  # B,3,N//2
        # mask_xyz_t1 = mask_points[:, :3, s:]

        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]

        if self.box_aware:
            pred_bc = seg_out[:, 2:, :] # B,9,N
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1) # B,3(+1)+9,N
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)
            # mask_xyz_t0 = mask_points[:, :, :N // 2]  # B,3,N//2
            # mask_xyz_t1 = mask_points[:, :, N // 2:]
        
        
        if self.use_memory_refine:
            memory_points = input_dict["memory_points"].transpose(1, 2)
            memory_bc = input_dict["memory_bc"].transpose(1, 2)
            bc_mask_xyz_t0 = torch.cat([mask_xyz_t0, candidate_bc[:, :, :N//2]], dim=1)
            bc_memory_points = torch.cat([memory_points, memory_bc], dim=1)
            memory_and_pre_points = torch.cat([bc_mask_xyz_t0, bc_memory_points], dim=2)
            pre_offset = self.pre_mlp(self.mini_pointnet(memory_and_pre_points))
            prev_boxes = torch.zeros_like(pre_offset)
            memory_points = points_utils.get_offset_points_tensor(memory_points.transpose(1, 2),
                                                                  prev_boxes[:, :4],
                                                                  pre_offset).transpose(1, 2)  # B,3,N//2
            # memory_points = torch.cat([memory_pc, memory_t], dim=1)
            if self.use_memory_cls:
                model_weights_path = "memoryclassification/" + self.category_name + "/best_model.pth"
                self.memory_cls.load_state_dict(torch.load(model_weights_path))
                self.memory_cls.eval()
                with torch.no_grad():
                    prediction_ori = self.memory_cls(mask_xyz_t0)
                    tmp = torch.cat([mask_xyz_t0, memory_points], dim=2)
                    sampled_indices = np.random.choice(2048, size=1024, replace=False)
                    combine = tmp[:, :, sampled_indices]
                    prediction_new = self.memory_cls(combine)
                    for i in range(mask_xyz_t0.shape[0]):
                        if prediction_new[i] > prediction_ori[i]:
                            mask_xyz_t0[i] = combine[i]
            else:
                mask_xyz_t0 = torch.cat([mask_xyz_t0, memory_points], dim=2)
            
            memory_box = pre_offset
            # output_dict["estimation_boxes_prev"] = pre_offset[:, :4]

        refinetime = time.time() - starttime

        # point_feature = self.origin_pointnet(mask_points) # m2track

        search_area = input_dict["ref_box_size"]
        w = search_area[:, 0]
        search_area[:, 0] = search_area[:, 1]
        search_area[:, 1] = w
        
        template_xyz, template_feature = self.backbone_net(mask_xyz_t0, [512, 256, 128]) # [1,64,1024]
        search_xyz, search_feature = self.backbone_net(mask_xyz_t1, [512, 256, 128])
        
        prev_feature = self.voxelization(template_feature, template_xyz, search_area) # [1,64,20,20,20]
        curr_feature = self.voxelization(search_feature, search_xyz, search_area)
        
        featuretime = time.time() - starttime
        
        # use mask
        # device = prev_feature.device
        # voxelmask = self.voxel_mask(search_area, device).permute(0, 3, 1, 2)
        # voxelmask = voxelmask.reshape(voxelmask.shape[0], voxelmask.shape[1], voxelmask.shape[2]*voxelmask.shape[3])
        # prev_feature = prev_feature + voxel_mask_prev_feature

        prev_feature = prev_feature.permute(0, 1, 4, 3, 2).contiguous()
        curr_feature = curr_feature.permute(0, 1, 4, 3, 2).contiguous()
        cml_prev = self.conv_middle_layers(prev_feature)
        cml_curr = self.conv_middle_layers(curr_feature)
        
        convtime = time.time() - starttime
        
        # cml_curr = cml_curr + voxelmask
        curr_feature = self.attention(template_xyz, cml_prev, search_xyz, cml_curr) # 用attention代替relationnet和xcorr
        point_feature = self.xcorr(curr_feature)
    
        motion_pred = self.motion_mlp(point_feature)  # B,4
        
        firsttime = time.time() - starttime

        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # previous bb, B,4
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        if self.use_memory_refine:
            # aux_box = points_utils.get_offset_box_tensor(memory_box, motion_pred_masked)
            prev_boxes = memory_box
        # else:
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # if self.use_memory_refine_after:
        #     memory_points = input_dict["memory_points"].transpose(1, 2)
        #     memory_bc = input_dict["memory_bc"].transpose(1, 2)
            
        #     memory_points = torch.cat([memory_points, mask_xyz_t0], dim=-1)
        #     memory_points = points_utils.get_offset_points_tensor(memory_points.transpose(1, 2),
        #                                                           prev_boxes[:, :4],
        #                                                           motion_pred_masked).transpose(1, 2)  # B,3,N//2
            
        #     mask_xyz_t01 = torch.cat([memory_points, mask_xyz_t1], dim=-1)
        #     # transform to the aux_box coordinate system
        #     mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
        #                                                                aux_box).transpose(1, 2)
            
        #     if self.box_aware:
        #         mask_bc = torch.cat([memory_bc, mask_pred_bc], dim=-1)
        #         mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_bc], dim=1)

        #     output_offset = self.pre_mlp(self.mini_pointnet(mask_xyz_t01))  # B,4
        #     # source code end
        #     output = points_utils.get_offset_box_tensor(aux_box, output_offset)
        #     output_dict["estimation_boxes"] = output
        # else:
        #     output_dict["estimation_boxes"] = aux_box
            

        # 2nd stage refinement
        if self.use_origin_second_stage:
            mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
            mask_xyz_t1 = mask_points[:, :3, N // 2:]

            # source code begin
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)
            
            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)

            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            # source code end
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            output_dict["estimation_boxes"] = output
        else:
            output_dict["estimation_boxes"] = aux_box

        secondtime = time.time() - starttime

        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            'segtime': segtime,
                            'refinetime': refinetime,
                            'featuretime': featuretime,
                            'convtime': convtime,
                            'firsttime': firsttime,
                            'secondtime': secondtime
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            seg_label = data['seg_label']
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label']
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = torch.sin(box_label_prev[:, 3])
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        if self.use_second_stage:
            estimation_boxes = output['estimation_boxes']  # B,4
            loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
            loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
            loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
            loss_dict["loss_center"] = loss_center
            loss_dict["loss_angle"] = loss_angle
        if self.use_prev_refinement:
            estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
            loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
            loss_angle_prev = F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)
            loss_total += (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
            loss_dict["loss_center_prev"] = loss_center_prev
            loss_dict["loss_angle_prev"] = loss_angle_prev

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)

        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight)

        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss


