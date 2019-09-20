import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch

from Utils.Transforms import RandomHorizontalFlip
from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter, Resize
from torch.utils.data import DataLoader
from Dataset.CrowdHumanGenerator import CrowdHumanGenerator
from Utils.Engine import train_one_epoch, evaluate

from Utils import Transforms as T
from Utils import utils
from Models.Efficient import efficientnet
from torchvision.models.mobilenet import mobilenet_v2
import torch.nn.functional as F


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def collate_fn(batch):
    return tuple(zip(*batch))






class LightHead(torch.nn.Module):
    def __init__(self, in_, backbone, mode="L"):
        super(LightHead, self).__init__()
        self.backbone = backbone
        if mode == "L":
            self.out = 256
        else:
            self.out = 64
        self.conv1 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out, out_channels=10*7*7, kernel_size=(1, 15),  stride=1, padding=(0, 7), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=self.out, out_channels=10*7*7, kernel_size=(1, 15), stride=1, padding=(0, 7), bias=True)

    def forward(self, input):
        x_backbone = self.backbone(input)
        x = self.conv1(x_backbone)
        x = self.relu(x)
        x = self.conv2(x)
        x_relu_2 = self.relu(x)

        x = self.conv3(x_backbone)
        x = self.relu(x)
        x = self.conv4(x)
        x_relu_4 = self.relu(x)

        return x_relu_2 + x_relu_4


# load a pre-trained model for classification and return
# only the features
backbone = efficientnet(net="B0", pretrained=True).features
backbone = LightHead(1280, backbone=backbone)

backbone.out_channels = 490


# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))


# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)


# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, min_size=800, max_size=1200, box_roi_pool=roi_pooler, box_detections_per_img=200)