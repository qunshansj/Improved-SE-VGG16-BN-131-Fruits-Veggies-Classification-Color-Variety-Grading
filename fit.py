python

# SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 两个全连接层，分别进行降维和升维
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # 加权

# SE-VGGConv模块
class SEVGGConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SEVGGConv, self).__init__()
        # 卷积 + BN + ReLU + SE
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x

# SE-VGG16-BN网络
class SEVGG16BN(nn.Module):
    def __init__(self, num_classes=131):
        super(SEVGG16BN, self).__init__()
        # 根据VGG16结构定义网络层
        self.features = nn.Sequential(
            # 输入 224x224x3
            SEVGGConv(3, 64, kernel_size=3, padding=1),
            SEVGGConv(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 56x56x128
            SEVGGConv(64, 128, kernel_size=3, padding=1),
            SEVGGConv(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 28x28x256
            SEVGGConv(128, 256, kernel_size=3, padding=1),
            SEVGGConv(256, 256, kernel_size=3, padding=1),
            SEVGGConv(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 14x14x512
            SEVGGConv(256, 512, kernel_size=3, padding=1),
            SEVGGConv(512, 512, kernel_size=3, padding=1),
            SEVGGConv(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 7x7x512
            SEVGGConv(512, 512, kernel_size=3, padding=1),
            SEVGGConv(512, 512, kernel_size=3, padding=1),
            SEVGGConv(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),  # 1x1x4096
        )
        # 定义分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 多损失函数融合
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # 融合系数
        self.cross_entropy = nn.CrossEntropyLoss()
        # 中心损失可以用triplet loss或者contrastive loss代替

    def forward(self, outputs, labels):
        # 此处需要根据具体情况添加中心损失的计算
        loss = self.cross_entropy(outputs, labels)
        return loss

# 封装为类
class SEVGG16BNCombinedLoss(nn.Module):
    def __init__(self, num_classes=131, alpha=0.1):
        super(SEVGG16BNCombinedLoss, self).__init__()
        self.model = SEVGG16BN(num_classes)
        self.loss = CombinedLoss(alpha)

    def forward(self, x, labels):
        outputs = self.model(x)
        loss = self.loss(outputs, labels)
        return loss
