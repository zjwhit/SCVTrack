import torch
import torch.nn as nn
import numpy as np
from pointnet_cls import PointNet


model = PointNet(input_channel=3, per_point_mlp=[64, 128, 256, 512], hidden_mlp=[512, 256], output_size=1)

model_weights_path = "best_model.pth"

model.load_state_dict(torch.load(model_weights_path))

data_path = '/...'

data = np.load(data_path)

input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

model.eval()

with torch.no_grad():
    outputs = model(input_data)

print("results:", outputs)
