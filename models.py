import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    # instantiates model's layers and loads any data artifacts required
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # where all the computation occurs
    # input is passed through the network of layers and various functions and then finally generates an output
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F. relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features


if __name__ == '__main__':
    net = LeNet()
    print(net)

    input = torch.rand(1, 1, 32, 32)
    print("\nImage batch shape:")
    print(input.shape)

    output = net(input)
    print("\nRaw output:")
    print(output)
    print(output.shape)