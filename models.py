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

    def __init__(self, model_type, pretrained=True):
        super(DepthEstimation, self).__init__()
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=pretrained)

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

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
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

    def __init__(self, object_detector, depth_estimator):
        super(Concater, self).__init__()

        self.depth_estimator = depth_estimator
        self.object_detector = object_detector
        self.resizer = torchvision.transforms.Resize([384, 512])
        self.object_detector.eval()
        self.depth_estimator.eval()


    def out(self, img):
        with torch.no_grad():
            prediction_OD = self.object_detector([img])
            
            input_batch = self.resizer(img)
            input_batch = torch.unsqueeze(input_batch, 0)
            prediction_DD = self.depth_estimator(input_batch)

            prediction_DD = torch.nn.functional.interpolate(
                prediction_DD.unsqueeze(1),
                size=img.shape[1:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        img = torch.as_tensor(img*255, dtype=torch.uint8)
        depth_percentage = []
        for i in range(10):
            center_x = int((prediction_OD[0]['boxes'][i,0]+prediction_OD[0]['boxes'][i,2])/2)
            center_y = int((prediction_OD[0]['boxes'][i,1]+prediction_OD[0]['boxes'][i,3])/2)
            depth_value = prediction_DD[center_y, center_x].numpy()
            if depth_value > 1.5*torch.mean(prediction_DD).numpy():
                label = 'Close'
            elif depth_value < 0.5*torch.mean(prediction_DD).numpy():
                label = 'Far'
            else:
                label = 'Middle'
            depth_percentage.append('{0:.2f} - {1}'.format(depth_value,label))
        plot_img = torchvision.utils.draw_bounding_boxes(img, prediction_OD[0]['boxes'][0:10,:], colors="red", labels=depth_percentage, font_size=24)
        return plot_img.permute(1,2,0).cpu().numpy()



