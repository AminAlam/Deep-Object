import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class DepthEstimation(nn.Module):
    '''
    Class performs Depth Estimation.
    '''

    def __init__(self, model_type):
        super(DepthEstimation, self).__init__()
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)

    def forward(self, x):
        out = self.model(x)
        return out

class ObjectDetector(nn.Module):
    '''
    Class performs Object Detection.
    '''
    def __init__(self, num_classes):
        super(ObjectDetector, self).__init__()

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
        self.model = model
        

    def forward(self, x1, x2=None):
        if x2!=None:
            out = self.model(x1, x2)
        else:
            out = self.model(x1)
        return out

class Concater(nn.Module):
    '''
    Class concats DepthEstimation and ObjectDetector
    '''

    def __init__(self):
        super(Concater, self).__init__()

        self.Depth = DepthEstimation()
        self.ObjectDetector = ObjectDetector()

    def forward(self, images):
        pass