import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.widgets
import matplotlib.patches
import mpl_toolkits.axes_grid1
from PIL import Image

from Home import *

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
import torchvision
import torchvision.transforms as transforms
import models
import utils

OD_model_save_dir = "Models/ODModel.pth.tar"
DD_model_save_dir = "Models/DDModel.pth.tar"

num_classes = 894
lr_OD = 0.005
lr_DD = 0.05
object_detector = models.ObjectDetector(num_classes=num_classes)
object_detector.eval()
params = [p for p in object_detector.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr_OD,
                            momentum=0.9, weight_decay=0.0005)

utils.load_checkpoint(torch.load(OD_model_save_dir, map_location=torch.device('cpu')), object_detector, optimizer)



model_type = "DPT_Hybrid"
depth_estimator = models.DepthEstimation(model_type, pretrained=False)
depth_estimator.eval()
params = [p for p in depth_estimator.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr_DD,
                            momentum=0.9, weight_decay=0.0005)
utils.load_checkpoint(torch.load(DD_model_save_dir, map_location=torch.device('cpu')), depth_estimator, optimizer)




class Main(QtWidgets.QMainWindow, Ui_HomeWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_HomeWindow.__init__(self)
        self.setupUi(self)
        self.load_image.clicked.connect(self.LoadImage)
        self.figure_images = Figure()
        self.figure_detected_images = Figure()
        self.ax1 = self.figure_images.add_subplot(131)
        self.ax2 = self.figure_images.add_subplot(132)
        self.ax3 = self.figure_images.add_subplot(133)
        self.ax4 = self.figure_detected_images.add_subplot(111)
        self.canvas1 = FigureCanvas(self.figure_images)
        self.canvas2 = FigureCanvas(self.figure_detected_images)
        

        self.ax1.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        self.ax2.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        self.ax3.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
        self.ax4.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)

        self.lay1 = QtWidgets.QVBoxLayout(self.loaded_image_widget)
        self.lay1.addWidget(self.canvas1)

        self.lay2 = QtWidgets.QVBoxLayout(self.detected_image_widget)  
        self.lay2.addWidget(self.canvas2)


        self.image_slider.hide()

        ## DD stuff
        self.resizer = torchvision.transforms.Resize([384, 512])

    def Slider_Init(self, num_images):
        self.image_slider.show()
        self.image_slider.setMaximum(num_images-1)
        self.image_slider.valueChanged.connect(self.plot_images)

    def LoadImage(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            files, _ = QFileDialog.getOpenFileNames(self, "Load File", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
            self.images = []
            for file in files:
                self.images.append(Image.open(file).convert("RGB"))
            self.Slider_Init(len(self.images))
            self.OD()
            self.DD()
            self.OD_DD()
            self.plot_images()
        except:
            pass

    def OD(self):
        self.objects_list = []
        for img in self.images:
            img = transforms.ToTensor()(img)
            with torch.no_grad():
                prediction = object_detector([img])
            print(prediction[0]['labels'][0:10])
            # for obj in range(prediction[0]['masks'].shape[0]):
            #     plot_img = plot_img + prediction[0]['masks'][obj, 0].mul(255)
            img = torch.as_tensor(img*255, dtype=torch.uint8)
            plot_img = torchvision.utils.draw_bounding_boxes(img, prediction[0]['boxes'][0:10,:], colors="red")
            self.objects_list.append(plot_img.permute(1,2,0).cpu().numpy())

    def DD(self):
        self.depth_list = []
        for img in self.images:
            img = transforms.ToTensor()(img)
            input_batch = self.resizer(img)
            input_batch = torch.unsqueeze(input_batch, 0)
            with torch.no_grad():
                prediction_DD = depth_estimator(input_batch)

            prediction_DD = torch.nn.functional.interpolate(
                prediction_DD.unsqueeze(1),
                size=img.shape[1:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            self.depth_list.append(prediction_DD.cpu().numpy())
        
    def OD_DD(self):
        self.depth_object_list = []
        for ww, img in enumerate(self.images):
            img = transforms.ToTensor()(img)
            input_batch = self.resizer(img)
            input_batch = torch.unsqueeze(input_batch, 0)

            with torch.no_grad():
                prediction_OD = object_detector([img])
                prediction_DD = depth_estimator(input_batch)

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
            plot_img = torchvision.utils.draw_bounding_boxes(img, prediction_OD[0]['boxes'][0:10,:], colors="red", labels=depth_percentage, font_size=14)
            plt.imsave("Final_DO_test_{}.png".format(ww+1), plot_img.permute(1,2,0).cpu().numpy())
            self.depth_object_list.append(plot_img.permute(1,2,0).cpu().numpy())


    def plot_images(self):

        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3.axis('off')
        self.ax4.axis('off')
        self.ax1.set_title('Main Image')
        self.ax2.set_title('Detected Objects')
        self.ax3.set_title('Estimated Depths')
        self.ax4.set_title('Joint Object Detection and Depth Estimation')
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        i = self.image_slider.value()
        self.ax1.imshow(self.images[i])
        self.ax2.imshow(self.objects_list[i])
        self.ax3.imshow(self.depth_list[i])
        self.ax4.imshow(self.depth_object_list[i])

        self.canvas1.draw()
        self.canvas2.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()

    sys.exit(app.exec_())