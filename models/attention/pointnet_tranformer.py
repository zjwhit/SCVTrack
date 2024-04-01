import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .transformer import TransformerDecoder, TransformerEncoder


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    # def __init__(self, input_channel=3, num_pos_feats=128):
    #     super().__init__()
    #     self.position_embedding_head = nn.Sequential(
    #         nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
    #         nn.BatchNorm1d(num_pos_feats),
    #         nn.ReLU(inplace=True),
    #         nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    # def forward(self, xyz):
    #     # xyz : BxNx3
    #     xyz = xyz.transpose(1, 2).contiguous()
    #     # Bx3xN
    #     position_embedding = self.position_embedding_head(xyz)
    #     return position_embedding
    def __init__(self, d_model):
        super().__init__()
        self.x_size = 20
        self.y_size = 20
        self.d_model = d_model
        
        # Calculate the total number of positions in the 2D grid
        total_positions = self.x_size * self.y_size

        # Create an embedding layer with total_positions as the input size and d_model as the output size
        self.embedding = nn.Embedding(total_positions, d_model)

    def forward(self, xy):
        x, y = xy[..., 0], xy[..., 1]

        # Calculate the position indices in the 2D grid
        positions = torch.arange(self.x_size)[:, None] * self.y_size + torch.arange(self.y_size)[None, :]
        positions = positions.view(-1)  # Flatten the positions to a 1D tensor

        # Convert the 2D coordinates (x, y) into 1D indices
        indices = x * self.y_size + y

        # Use nn.Embedding to get the positional embeddings for the given indices
        positional_embeddings = self.embedding(indices)

        # Reshape the positional embeddings to [B, d_model, 20*20]
        positional_embeddings = positional_embeddings.permute(0, 2, 1)

        return positional_embeddings


class PointnetTransformerSiamese(nn.Module):
    def __init__(self, num_layers, channel):
        super(PointnetTransformerSiamese, self).__init__()

        d_model = channel
        # num_layers = num_layers
        self.with_pos_embed = False

        # self.FC_layer_cla = (
        #         pt_utils.Seq(256)
        #         .conv1d(256, bn=True)
        #         .conv1d(256, bn=True)
        #         .conv1d(1, activation=None))
        # self.fea_layer = (pt_utils.Seq(256)
        #         .conv1d(256, bn=True)
        #         .conv1d(256, activation=None))


        multihead_attn = MultiheadAttention(
            feature_dim=d_model, n_head=1, key_feature_dim=64)

        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(d_model)
            decoder_pos_embed = PositionEmbeddingLearned(d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None

        self.encoder = TransformerEncoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_encoder_layers=num_layers,
            self_posembed=encoder_pos_embed)
        self.decoder = TransformerDecoder(
            multihead_attn=multihead_attn, FFN=None,
            d_model=d_model, num_decoder_layers=num_layers,
            self_posembed=decoder_pos_embed)
        

    
    def transform_fuse(self, prev_xyz, prev_feature, curr_xyz, curr_feature):
        """Use transformer to fuse feature.

        template_feature : BxCxN
        template_xyz : Bx3xN
        """
        # BxCxN -> NxBxC
        B, _, _ = curr_feature.shape
        curr_feature = curr_feature.permute(2, 0, 1)
        prev_feature = prev_feature.permute(2, 0, 1)
        curr_xyz = curr_xyz.permute(0, 2, 1) # B,3,N-->B,N,3
        prev_xyz = prev_xyz.permute(0, 2, 1)

        # num_img_train = search_feature.shape[0]
        # num_img_template = template_feature.shape[0]

        ## encoder
        device = curr_feature.device
        xy_vector = torch.stack(torch.meshgrid(torch.arange(20), torch.arange(20)), dim=-1).view(-1, 2).to(device)
        
        input_xy = torch.unsqueeze(xy_vector, 0).expand(B, -1, -1)

        encoded_memory = self.encoder(prev_feature,
                                      query_pos=input_xy if self.with_pos_embed else None)

        encoded_feat = self.decoder(curr_feature,
                                    memory=encoded_memory,
                                    query_pos=input_xy if self.with_pos_embed else None)

        # NxBxC -> BxCxN
        encoded_feat = encoded_feat.permute(1, 2, 0)
        # encoded_feat = self.fea_layer(encoded_feat)

        return encoded_feat

    def forward(self, prev_xyz, prev_feature, curr_xyz, curr_feature):
        """
            prev_frame: B*256*S 
            curr_frame: B*256*S
        """

        fusion_feature = self.transform_fuse(
            prev_xyz, prev_feature, curr_xyz, curr_feature)

        # attention_feature = self.mlp(fusion_feature)   

        return fusion_feature
