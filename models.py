import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DepthEstimation(nn.Module):
    '''
    Class performs Depth Estimation.
    '''

    def __init__(self):
        super(DepthEstimation, self).__init__()

    def forward(self, images):
        pass

class ObjectDetector(nn.Module):
    '''
    Class performs Object Detection.
    '''

    def __init__(self):
        super(ObjectDetector, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=32, stride=2, padding=1) # out: 4 x 305 x 305
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=15, stride=2, padding=1) # out: 8 x 147 x 147
        self.maxPol1 = nn.MaxPool2d(kernel_size=9, stride=2, padding=1) # out: 8 x 71 x 71
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, stride=2, padding=1) # out: 16 x 32 x 32
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.maxPol2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1) # out: 32 x 8 x 8
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1) # out: 64 x 4 x 4
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=1) # out: 128 x 1 x 1
        self.fc1 = nn.Linear(in_features=128, out_features=512) # out: 512
        self.fc2 = nn.Linear(in_features=512, out_features=1024) # out: 1024
        # 1024 reshaped to 1*32*32
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1) # out: 1 x 63 x 63
        self.deconv2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=4) # out: 1 x 119 x 119
        self.deconv3 = nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=5, stride=2, padding=1) # out: 2 x 239 x 239
        self.deconv4 = nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=6, stride=3, padding=40) # out: 3 x 640 x 640

        self.drop_out = nn.Dropout(0.5)
        

    def forward(self, x):
        batch_size = x.shape[0]
        ### encoding
        x = self.conv1(x)
        x = self.drop_out(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.maxPol1(x)

        x = self.conv3(x)
        x = self.drop_out(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = self.maxPol2(x)

        x = self.conv5(x)
        x = self.drop_out(x)
        x = self.conv6(x)
        x = F.relu(x)
        
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = F.relu(x)
    
        x = torch.reshape(x, (batch_size,1,32,32))
        
        ### decoding
        x = self.deconv1(x)
        x = self.drop_out(x)
        x = F.relu(x)

        x = self.deconv2(x)
        x = self.drop_out(x)
        x = F.relu(x)

        x = self.deconv3(x)
        x = self.drop_out(x)
        x = F.relu(x)

        out = self.deconv4(x)
        x = F.relu(x)
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