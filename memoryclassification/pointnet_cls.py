import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, category_name, test_ratio=0.1, test=False):
        self.test = test
        self.root_dir = root_dir
        self.category_name = category_name
        self.positive_dir = os.path.join(root_dir, category_name)
        self.negative_dir = os.path.join(root_dir, 'negative', category_name)
        self.positive_file_list = os.listdir(self.positive_dir)
        self.negative_file_list = os.listdir(self.negative_dir)

        self.test_ratio = test_ratio
        self.num_positive_samples = len(self.positive_file_list)
        self.num_negative_samples = len(self.negative_file_list)
        
        self.num_positive_test = int(self.num_positive_samples * test_ratio)
        self.num_negative_test = int(self.num_negative_samples * test_ratio)
        
        self.positive_train_list = self.positive_file_list[self.num_positive_test:]
        self.negative_train_list = self.negative_file_list[self.num_negative_test:]
        
        self.positive_test_list = self.positive_file_list[:self.num_positive_test]
        self.negative_test_list = self.negative_file_list[:self.num_negative_test]

    def __len__(self):
        if not self.test:
            return len(self.positive_train_list) + len(self.negative_train_list)
        else:
            return len(self.positive_test_list) + len(self.negative_test_list)

    def __getitem__(self, idx):
        if not self.test:
            if idx < len(self.positive_train_list):
                file_name = self.positive_train_list[idx]
                file_path = os.path.join(self.positive_dir, file_name)
                label = 1  
            else:
                file_name = self.negative_train_list[idx - len(self.positive_train_list)]
                file_path = os.path.join(self.negative_dir, file_name)
                label = 0  
        else:
            if idx < len(self.positive_test_list):
                file_name = self.positive_test_list[idx]
                file_path = os.path.join(self.positive_dir, file_name)
                label = 1  
            else:
                file_name = self.negative_test_list[idx - len(self.positive_test_list)]
                file_path = os.path.join(self.negative_dir, file_name)
                label = 0  

        point_cloud = np.load(file_path)

        sample_size = 1024
        num_points = point_cloud.shape[1]
        new_pts_idx = None
        if num_points > 2:
            new_pts_idx = np.random.choice(num_points, size=sample_size, replace=sample_size > num_points)
        # else:
        #     new_pts_idx = np.random.choice(num_points, size=sample_size, replace=True)

        if new_pts_idx is not None:
            point_cloud = point_cloud[:, new_pts_idx]
        else:
            point_cloud = np.zeros((3, sample_size), dtype='float32')
            
        return point_cloud, label

    def get_test_data(self):
        test_data = []
        for file_name in self.positive_test_list:
            file_path = os.path.join(self.positive_dir, file_name)
            point_cloud = np.load(file_path)
            test_data.append((point_cloud, 1))
        
        for file_name in self.negative_test_list:
            file_path = os.path.join(self.negative_dir, file_name)
            point_cloud = np.load(file_path)
            test_data.append((point_cloud, 0))
        
        return test_data


class PointNet(nn.Module):
    def __init__(self, input_channel, per_point_mlp, hidden_mlp, output_size=0):
        super(PointNet, self).__init__()
        seq_per_point = []
        in_channel = input_channel
        for out_channel in per_point_mlp:
            seq_per_point.append(nn.Conv1d(in_channel, out_channel, 1))
            seq_per_point.append(nn.BatchNorm1d(out_channel))
            seq_per_point.append(nn.ReLU())
            in_channel = out_channel
        seq_hidden = []
        for out_channel in hidden_mlp:
            seq_hidden.append(nn.Linear(in_channel, out_channel))
            seq_hidden.append(nn.BatchNorm1d(out_channel))
            seq_hidden.append(nn.ReLU())
            in_channel = out_channel

        self.features = nn.Sequential(*seq_per_point,
                                      nn.AdaptiveMaxPool1d(output_size=1),
                                      nn.Flatten(),
                                      *seq_hidden)
        self.output_size = output_size
        self.fc = nn.Linear(in_channel, output_size)

    def forward(self, x):
        """
        :param x: B,C,N
        :return: B,output_size
        """
        x = self.features(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

def train(model, train_data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_data_loader):
        
        inputs, labels = batch
        inputs = inputs.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1)) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        predicted = (outputs >= 0.5).float()  
        total_correct += (predicted == labels.float().unsqueeze(1)).sum().item()
        total_samples += inputs.size(0)

        if batch_idx % 100 == 99:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print('Batch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2%}'
                  .format(batch_idx + 1, len(train_data_loader), avg_loss, accuracy))
    
    print(total_samples)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

def test(model, test_data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            
            total_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total_correct += (predicted == labels.float().unsqueeze(1)).sum().item()
            total_samples += inputs.size(0)

    print(total_samples)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


if __name__ == '__main__':
    category_name = 'Car'
    num_workers = 8
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    data_path = '/tmp_memory/'
    train_datasets = PointCloudDataset(root_dir=data_path, category_name=category_name, test=False)
    test_datasets = PointCloudDataset(root_dir=data_path, category_name=category_name, test=True)

    train_loader = DataLoader(train_datasets, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_datasets, batch_size=1, num_workers=num_workers, shuffle=False)

    model = PointNet(input_channel=3, per_point_mlp=[64, 128, 256, 512], hidden_mlp=[512, 256], output_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    best_model_path = category_name + "/best_model.pth"

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2%}'
              .format(epoch+1, num_epochs, train_loss, train_accuracy))
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
        print('Epoch [{}/{}], Test Loss: {:.4f}, Test Accuracy: {:.2%}'
              .format(epoch+1, num_epochs, test_loss, test_accuracy))
