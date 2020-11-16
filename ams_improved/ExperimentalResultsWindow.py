from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from PySide2.QtWidgets import QMessageBox
from PySide2 import QtWidgets
from pyqtgraph import PlotWidget, plot

from ui_designs.ExperimentalResultsWindow import Ui_ExperimentalResultsWindow

class Ui_ExperimentalResultsWindow_(QWidget, Ui_ExperimentalResultsWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def displayUi(self):
        self.show()

        # Initialise plotting window
        self.plotWidget.canvas.fig.tight_layout()

        # Initialise buttons

    def plot_results(self):
        pass

    def clear_graph(self):
        pass
