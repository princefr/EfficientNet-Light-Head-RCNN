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

train_dataset = CrowdHumanGenerator(path="./data", type="train", config=None, transform=get_transform(train=True))
validation_dataset = CrowdHumanGenerator(path="./data", type="validation", config=None, transform=get_transform(train=False))

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)




class LightHead(torch.nn.Module):
    def __init__(self, in_, backbone):
        super(LightHead, self).__init__()
        self.backbone = backbone
        self.conv1 = torch.nn.Conv2d(in_channels=in_, out_channels=256, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=10*7*7, kernel_size=(1, 15),  stride=1, padding=(0, 7), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=in_, out_channels=256, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=10*7*7, kernel_size=(1, 15), stride=1, padding=(0, 7), bias=True)

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
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, min_size=700, max_size=1100, box_roi_pool=roi_pooler)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_loader, device=device)
    torch.save(model.state_dict(), "efficient_rcnn_" + str(epoch) + ".pth")







