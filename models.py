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