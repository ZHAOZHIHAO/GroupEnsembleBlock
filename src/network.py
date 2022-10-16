import torch
import torch.nn as nn
import torch.nn.functional as F
from src.group_ensemble_block import GroupEnsembleBlock


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        print("Set gamma of last batchnorm layer in a residual block to 0")
        self.bn3.weight.data.fill_(0.0)
        self.shortcut = nn.Sequential()
        print("Avg pool and 1*1 conv in shortcut")
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                # nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        elif in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        else:
            print("Identity shortcut")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetCIFAR100(nn.Module):
    def __init__(self, block, num_blocks, paras):
        super(ResNetCIFAR100, self).__init__()
        embedding_dim = paras.embedding_dim

        # for the backbone
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # constructing projector
        proj_hid = 2048
        proj_hid2 = 2048
        proj_out = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(512*block.expansion, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(proj_hid, proj_hid2),
            nn.BatchNorm1d(proj_hid2),
            nn.ReLU(inplace=True),)
        print("proj_hid", proj_hid)
        print("proj_hid2", proj_hid2)

        # constructing predictor
        pred_hid = 2048
        pred_out = embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hid, pred_out),)
        print("pred_hid", pred_hid)

        self.ensembleBlock = GroupEnsembleBlock(input_length=2048, output_length=embedding_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # 512

        out = self.projection(out)
        out = self.ensembleBlock(out)
        out_predictor = self.predictor(out)
        out_projector = F.normalize(out, dim=-1)

        return out_projector, out_predictor

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks, paras):
        super(ResNetImageNet, self).__init__()
        embedding_dim = paras.embedding_dim

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # constructing projector
        proj_hid = 2048
        proj_hid2 = 2048
        proj_out = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(512*block.expansion, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(proj_hid, proj_hid2),
            nn.BatchNorm1d(proj_hid2),
            nn.ReLU(inplace=True),
        )

        # constructing predictor
        pred_hid = 2048
        pred_out = embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hid, pred_out),)
        print("pred_hid", pred_hid)

        self.ensembleBlock = GroupEnsembleBlock(input_length=2048, output_length=embedding_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).squeeze()

        out = self.projection(out)
        out = self.ensembleBlock(out)
        out_predictor = self.predictor(out)
        out_projector = F.normalize(out, dim=-1)

        return out_projector, out_predictor


def ResNet50CIFAR100(paras):
    return ResNetCIFAR100(Bottleneck, [3,4,6,3], paras)


def ResNet50ImageNet(paras):
    return ResNetImageNet(Bottleneck, [3,4,6,3], paras)