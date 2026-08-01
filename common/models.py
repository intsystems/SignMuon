import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2(nn.Module):
    """
    Two-block convolutional network, the federated experiments' backbone.

    Inputs:
    - MNIST (1x28x28)
    - CIFAR10 (3x32x32)

    Conv2d(in_channels, n_kernels, kernel_size=5, padding=2) 
        --> BatchNorm2d --> ReLU --> MaxPool2d(2)
    Conv2d(n_kernels, 2*n_kernels, kernel_size=5, padding=0) 
        --> BatchNorm2d --> ReLU --> MaxPool2d(2)
    Flatten
    Linear(flattened_dim, 120) --> BatchNorm1d --> ReLU
    Linear(120, out_dim)
    """
    def __init__(
            self, 
            in_channels=3,
            input_size=32,
            n_kernels=64,  
            out_dim=10,
    ):
        super().__init__()     

        self.conv1 = nn.Conv2d(in_channels, n_kernels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(n_kernels)

        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(2 * n_kernels)

        self.pool = nn.MaxPool2d(2, 2)

        conv1_out = (input_size - 5 + 2 * 2) // 1 + 1
        pool1_out = conv1_out // 2
        conv2_out = (pool1_out - 5 + 2 * 0) // 1 + 1
        pool2_out = conv2_out // 2
        flattened = 2 * n_kernels * pool2_out * pool2_out

        self.latent_dim = 120
        self.fc1 = nn.Linear(flattened, self.latent_dim)
        self.bn3 = nn.BatchNorm1d(self.latent_dim)
        self.classifier = nn.Linear(self.latent_dim, out_dim)

    def features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        return x

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    


class ResNet9(nn.Module):
    """
    ResNet-9 for low-resolution images.

    Inputs:
    - CIFAR10/100 (3x32x32)

    Conv2d(3, 64) -> BatchNorm2d -> ReLU
    Conv2d(64, 128) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
               + ResNet Block: [Conv2d(128, 128) -> BN -> ReLU] x 2
    Conv2d(128, 256) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    Conv2d(256, 512) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
               + ResNet Block: [Conv2d(512, 512) -> BN -> ReLU] x 2
    AdaptiveAvgPool2d(1) -> Flatten -> Linear(512, num_classes)
    """
    def __init__(
            self, 
            num_classes=10,
    ):
        super().__init__()

        def conv_block(in_c, out_c, pool=False):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.res2(x)
        return self.classifier(x)        
    



class BasicBlock(nn.Module):
    """
    The classical ResNet-18 residual block: two convolutions plus a skip
    connection.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # When the spatial size (stride != 1) or the channel count changes, a
        # 1x1 convolution reshapes the input so it can be added to the output.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip Connection
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) 
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out