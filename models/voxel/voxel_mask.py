import torch
import torch.nn as nn

class VoxelMaskEmbedding(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, divide, num_pos_feats=128):
        super().__init__()
        self.num_voxels_x, self.num_voxels_y, self.num_voxels_z = divide
        self.embedding_dim = num_pos_feats
        self.voxel_mask_embed = nn.Embedding(3, num_pos_feats)
        self.voxel_mask_embed.weight.data = torch.tensor([[0, 0.5, 1]] * num_pos_feats).transpose(0, 1)

    def forward(self, search_area, device):
        search_area.to(device)
        B = search_area.size(0)  # Get the batch size
        l, w, h = search_area[:, 0], search_area[:, 1], search_area[:, 2]
        x, y, z = [(l-4)/1.25, (w-4)/1.25, (h-4)/1.25]
        
        # print("Embedding")
        # print(self.voxel_mask_embed.weight)
        # print(torch.mean(self.voxel_mask_embed.weight[0]))
        # print(torch.mean(self.voxel_mask_embed.weight[1]))
        # print(torch.mean(self.voxel_mask_embed.weight[2]))
        # print(torch.var(self.voxel_mask_embed.weight[0]))
        # print(torch.var(self.voxel_mask_embed.weight[1]))
        # print(torch.var(self.voxel_mask_embed.weight[2]))

        embedding_arrays = []
        for b in range(B):
            # Generate voxel coordinates for each search_area in the batch
            voxel_coords = torch.meshgrid(
                torch.linspace(-l[b]/2 + l[b]/(2*self.num_voxels_x), l[b]/2 - l[b]/(2*self.num_voxels_x), self.num_voxels_x),
                torch.linspace(-w[b]/2 + w[b]/(2*self.num_voxels_y), w[b]/2 - w[b]/(2*self.num_voxels_y), self.num_voxels_y),
                torch.linspace(-h[b]/2 + h[b]/(2*self.num_voxels_z), h[b]/2 - h[b]/(2*self.num_voxels_z), self.num_voxels_z)
            )
            
            voxel_coords_flat = torch.stack(voxel_coords).flatten().unsqueeze(1).float()
            voxel_center_coords = voxel_coords_flat.view(3, -1).transpose(0, 1).to(device)
            # Initialize embedding array for the current search_area
            embedding_array = torch.zeros((self.num_voxels_x, self.num_voxels_y, self.num_voxels_z, self.embedding_dim)).to(device)
            
            center_region_mask = (
                (voxel_center_coords[:, 0].abs() < x[b] / 2 - 0.5 * (l[b] / self.num_voxels_x)) &
                (voxel_center_coords[:, 1].abs() < y[b] / 2 - 0.5 * (w[b] / self.num_voxels_y)) &
                (voxel_center_coords[:, 2].abs() < z[b] / 2 - 0.5 * (h[b] / self.num_voxels_z))
            )
            boundary_mask = (
                (~center_region_mask) &
                (voxel_center_coords[:, 0].abs() < x[b] / 2 + 0.5 * (l[b] / self.num_voxels_x)) &
                (voxel_center_coords[:, 1].abs() < y[b] / 2 + 0.5 * (w[b] / self.num_voxels_y)) &
                (voxel_center_coords[:, 2].abs() < z[b] / 2 + 0.5 * (h[b] / self.num_voxels_z))
            )

            center_region_mask = center_region_mask.view(self.num_voxels_x, self.num_voxels_y, self.num_voxels_z)
            boundary_mask = boundary_mask.view(self.num_voxels_x, self.num_voxels_y, self.num_voxels_z)

            embedding_array[center_region_mask] = self.voxel_mask_embed.weight[2]
            embedding_array[boundary_mask] = self.voxel_mask_embed.weight[1]
            embedding_array[~(center_region_mask | boundary_mask)] = self.voxel_mask_embed.weight[0]

            embedding_array = embedding_array[:, :, self.num_voxels_z // 2]
            embedding_arrays.append(embedding_array.unsqueeze(0))  # Add the processed embedding for this search_area to the list
            
        return torch.cat(embedding_arrays, dim=0)  # Concatenate the processed embeddings for all search_areas in the batch

# # Test the implementation
# if __name__ == "__main__":
#     # Create a sample input search_area
#     search_area = torch.tensor([[9, 9, 9], [9, 9, 9]], dtype=torch.float32)

#     # Create an instance of the VoxelMaskEmbedding class
#     divide = (4, 4, 4)
#     model = VoxelMaskEmbedding(divide)

#     # Perform the forward pass
#     device = torch.device("cpu")
#     output = model(search_area, device)

#     print("Output shape:", output.shape)
#     print("Output (XY-plane mask) for the first search area:")
#     print(output[0])