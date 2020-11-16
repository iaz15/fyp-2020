from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from PySide2.QtWidgets import QMessageBox
from PySide2 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import threading
import numpy as np
import sys
import os
import plotly
import plotly.express as px

from PySide2 import QtCore, QtWidgets, QtWebEngineWidgets
import plotly.express as px
import plotly.graph_objects as go

# https://www.ics.com/blog/we-ported-qt-app-c-python-heres-what-happened

# Generate the UI Python Files
print("Generating the UI")
print("|-- Creating Main Window")
os.system("pyside2-uic ui_designs\\MainWindow.ui -o ui_designs\\MainWindow.py")
print("|-- Creating Test Conditions Window")
os.system("pyside2-uic ui_designs\\TestConditionsWindow.ui -o ui_designs\\TestConditionsWindow.py")
print("|--> Finished!\n")

# Once update UI has been generated, import the files
from ui_designs.MainWindow import Ui_MainWindow

from TestConditionsWindow import Ui_TestConditionsWindow_

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.diplayUi()

        self.testConditionsWindow = Ui_TestConditionsWindow_()

    def diplayUi(self):
        # On click events
        self.addTestConditionsBtn.clicked.connect(self.launch_test_conditions_window)

    def launch_test_conditions_window(self):
        print("Launching Test Conditions Window")
        self.testConditionsWindow.displayUi()

    def launch_experimental_results_window(self):
        print("Launching Experimental Results Window")
        self.experimentalResultsWindow.displayUi()


def main():
    """ Main Function """
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    main_window = MainWindow()
    main_window.show()

    app.exec_()

if __name__ == "__main__":

    main()
    pass

# Useful Resources
# https://python-forum.io/Thread-access-variables-between-classes - Passing information between classes
# https://www.youtube.com/watch?v=hZe5hzQY8ow - Data visualisations
