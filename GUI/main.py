import sys
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


from Home import *


class Main(QtWidgets.QMainWindow, Ui_HomeWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_HomeWindow.__init__(self)
        self.setupUi(self)
        self.load_image.clicked.connect(self.LoadImage)
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
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
            self.images.append(plt.imread(file))
        self.Slider_Init(len(self.images))
        self.plot_images()
            
    def plot_images(self):
        self.ax.clear()
        i = self.image_slider.value()
        self.ax.imshow(self.images[i])
        self.canvas.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()

    sys.exit(app.exec_())