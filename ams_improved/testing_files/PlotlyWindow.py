from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from PySide2.QtWidgets import QMessageBox
from PySide2 import QtWidgets
from pyqtgraph import PlotWidget, plot
import plotly
import plotly.express as px
from PySide2 import QtCore, QtWidgets, QtWebEngineWidgets

from ui_designs.PlotlyWindow import Ui_PlotlyWindow

class Ui_PlotlyWindow_(QWidget, Ui_PlotlyWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.browser.show()
        # self.show_graph()

    def displayUi(self):
        self.show()

    def show_graph(self):
        df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
        df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' # Represent only large countries
        fig = px.pie(df, values='pop', names='country', title='Population of European continent')
        # fig.show()

        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))