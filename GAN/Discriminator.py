import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Output : 6x6x10
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)# output : 6x6x20
        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout(p=0.2) # default p=0.5
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        self.pooling = nn.MaxPool2d(2,2)
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        #print(x.size()) # torch.Size([128, 10, 24, 24])
        x = self.pooling(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.dropout2d(x)
        #print(x.size()) # torch.Size([128, 20, 8, 8])
        x = self.pooling(x)
        x = self.act(x)
        #print(x.size()) # torch.Size([128, 20, 4, 4])
        x = self.flatten(x)
        #print(x.size()) # torch.Size([128, 320])
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # Return value between 0 or 1