import torch.nn as nn
import torch

architecture_config = [
    # (kernel size, number of filters, stride, padding)
    # M means max pool of (2, input fi)
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    (1,256,1,0),
    (3,512,1,1),
    (1,256,1,0),
    (3,512,1,1),
    (1,256,1,0),
    (3,512,1,1),
    (1,256,1,0),
    (3,512,1,1),
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    (1,512,1,0),
    (3,1024,1,1),
    (1,512,1,0),
    (3,1024,1,1),
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1)
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # bias is set false as batchnorm is implemented (?!)
        self.conv = nn.Conv2d(in_channels, out_channels, bias= False, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyRelu(self.batchNorm(self.conv(x))) 


class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._createConvLayers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _createConvLayers(self,architecture):
        layers = []
        in_channels = self.in_channels
        for layer in architecture:
            if type(layer) == tuple:
                layers += [CNNBlock(in_channels, \
                    layer[1], kernel_size=layer[0], stride=layer[2], padding=layer[3])]
                in_channels = layer[1]
            elif type(layer) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S = split_size
        B = num_boxes
        C = num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096,S*S*(C+ B*5))
        )


def test(S=7, B=2, C=20):
    model = YoloV1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)    

test()

