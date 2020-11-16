# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MatplotlibWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *

from mplwidget import MplWidget


class Ui_MatplotlibWindow(object):
    def setupUi(self, MatplotlibWindow):
        if not MatplotlibWindow.objectName():
            MatplotlibWindow.setObjectName(u"MatplotlibWindow")
        MatplotlibWindow.resize(572, 442)
        self.verticalLayout = QVBoxLayout(MatplotlibWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.plotWidget = MplWidget(MatplotlibWindow)
        self.plotWidget.setObjectName(u"plotWidget")
        self.plotWidget.setMinimumSize(QSize(550, 420))
        self.plotWidget.setAutoFillBackground(False)

        self.verticalLayout.addWidget(self.plotWidget)


        self.retranslateUi(MatplotlibWindow)

        QMetaObject.connectSlotsByName(MatplotlibWindow)
    # setupUi

    def retranslateUi(self, MatplotlibWindow):
        MatplotlibWindow.setWindowTitle(QCoreApplication.translate("MatplotlibWindow", u"Form", None))
    # retranslateUi

