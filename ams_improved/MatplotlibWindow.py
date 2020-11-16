from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from PySide2.QtWidgets import QMessageBox
from PySide2 import QtWidgets
from pyqtgraph import PlotWidget, plot
from PySide2 import QtCore

import os
from ui_designs.MatplotlibWindow import Ui_MatplotlibWindow


class MatplotlibWindow_(QWidget, Ui_MatplotlibWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def displayUi(self):
        self.show()