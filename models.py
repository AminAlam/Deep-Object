import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DepthEstimation(nn.Module):
    '''
    Class performs Depth.
    '''

    def __init__(self):
        super(DepthEstimation, self).__init__()

    def forward(self, images):


class ObjectDetector(nn.Module):
    '''
    Class performs Object Detection.
    '''

    def __init__(self):
        super(ObjectDetector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=capacity, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=capacity*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=capacity*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=capacity*2*7*7, out_features=latent_dims)

    def forward(self, images):


class Concater(nn.Module):
    '''
    Class concats DepthEstimation and ObjectDetector
    '''

    def __init__(self):
        super(Concater, self).__init__()

    self.Depth = DepthEstimation()
    self.ObjectDetector = ObjectDetector()

    def forward(self, images):