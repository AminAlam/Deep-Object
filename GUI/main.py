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
matplotlib.use("TkAgg")
from Home import *


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
object_detector = models.ObjectDetector(num_classes=num_classes)
object_detector.eval()
params = [p for p in object_detector.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr_OD,
                            momentum=0.9, weight_decay=0.0005)

utils.load_checkpoint(torch.load(OD_model_save_dir, map_location=torch.device('cpu')), object_detector, optimizer)



class Main(QtWidgets.QMainWindow, Ui_HomeWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_HomeWindow.__init__(self)
        self.setupUi(self)
        self.load_image.clicked.connect(self.LoadImage)
        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(131)
        self.ax2 = self.figure.add_subplot(132)
        self.ax3 = self.figure.add_subplot(133)
        self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)

        self.lay = QtWidgets.QVBoxLayout(self.loaded_image_widget)        
        # self.lay.addWidget(self.toolbar)
        self.lay.addWidget(self.canvas)
        self.image_slider.hide()

    def Slider_Init(self, num_images):
        self.image_slider.show()
        self.image_slider.setMaximum(num_images-1)
        self.image_slider.valueChanged.connect(self.plot_images)

    def LoadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Load File", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        self.images = []
        for file in files:
            self.images.append(Image.open(file).convert("RGB"))
        self.Slider_Init(len(self.images))
        self.OD()
        self.plot_images()

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

    def plot_images(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        i = self.image_slider.value()
        self.ax1.imshow(self.images[i])
        self.ax2.imshow(self.objects_list[i])

        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()

    sys.exit(app.exec_())