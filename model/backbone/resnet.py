import torchvision
import torch.nn as nn

__all__ = ['ResNet18', 'ResNet50','Semantic_ResNet18','Semantic_ResNet50']


class ResNet18(nn.Module):
    #output_size = 512

    def __init__(self,n_classes):
        super(ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
        self.fc = nn.Linear(512, n_classes)
    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = pool.view(pool.size(0), -1)
        out  = self.fc(pool)
        if get_ha:
            return b1, b2, b3, b4, pool, out

        return out


class ResNet50(nn.Module):
    #output_size = 2048

    def __init__(self, n_classes):
        super(ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
        self.fc = nn.Linear(2048, n_classes)
    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = pool.view(pool.size(0), -1)
        out  = self.fc(pool)
        if get_ha:
            return b1, b2, b3, b4, pool, out

        return out

class Semantic_ResNet18(nn.Module):
    #output_size = 512

    def __init__(self,n_classes):
        super(Semantic_ResNet18, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
        self.fc1 = nn.Linear(512, 300)
        self.fc2 = nn.Linear(300, n_classes)
    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = pool.view(pool.size(0), -1)
        semantic  = self.fc1(pool)
        out  = self.fc2(semantic)
        if get_ha:
            return b1, b2, b3, b4, semantic, out

        return out

class Semantic_ResNet50(nn.Module):
    #output_size = 2048

    def __init__(self,n_classes):
        super(Semantic_ResNet50, self).__init__()
        pretrained = torchvision.models.resnet50(pretrained=True)

        for module_name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            self.add_module(module_name, getattr(pretrained, module_name))
        self.fc1 = nn.Linear(2048, 300)
        self.fc2 = nn.Linear(300, n_classes)
    def forward(self, x, get_ha=False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)
        pool = self.avgpool(b4)
        pool = pool.view(pool.size(0), -1)
        semantic  = self.fc1(pool)
        out  = self.fc2(semantic)
        if get_ha:
            return b1, b2, b3, b4, semantic, out

        return out
