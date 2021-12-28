import sys
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvas

from Home import *

class Main(QtWidgets.QMainWindow, Ui_HomeWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_HomeWindow.__init__(self)
        self.setupUi(self)
        self.load_image.clicked.connect(self.LoadImage)

    def LoadImage(self):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            files, _ = QFileDialog.getOpenFileNames(self, "Load File", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
            images = []
            for file in files:
                images.append(plt.imread(file))
            print(images)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()

    sys.exit(app.exec_())