# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(201, 244)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setMaximumSize(QSize(16777215, 20))
        font = QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setInputMethodHints(Qt.ImhNone)
        self.label_10.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_10, 0, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.addTestConditionsBtn = QPushButton(self.centralwidget)
        self.addTestConditionsBtn.setObjectName(u"addTestConditionsBtn")
        font1 = QFont()
        font1.setPointSize(9)
        self.addTestConditionsBtn.setFont(font1)

        self.verticalLayout.addWidget(self.addTestConditionsBtn)

        self.averageResultsBtn = QPushButton(self.centralwidget)
        self.averageResultsBtn.setObjectName(u"averageResultsBtn")
        self.averageResultsBtn.setFont(font1)

        self.verticalLayout.addWidget(self.averageResultsBtn)

        self.modelFittingBtn_2 = QPushButton(self.centralwidget)
        self.modelFittingBtn_2.setObjectName(u"modelFittingBtn_2")
        self.modelFittingBtn_2.setFont(font1)

        self.verticalLayout.addWidget(self.modelFittingBtn_2)

        self.modelFittingBtn = QPushButton(self.centralwidget)
        self.modelFittingBtn.setObjectName(u"modelFittingBtn")
        self.modelFittingBtn.setFont(font1)

        self.verticalLayout.addWidget(self.modelFittingBtn)

        self.visualiseResultsBtn = QPushButton(self.centralwidget)
        self.visualiseResultsBtn.setObjectName(u"visualiseResultsBtn")
        self.visualiseResultsBtn.setFont(font1)

        self.verticalLayout.addWidget(self.visualiseResultsBtn)


        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Tribo-Mate", None))
        self.addTestConditionsBtn.setText(QCoreApplication.translate("MainWindow", u"Add Experimental Tests", None))
        self.averageResultsBtn.setText(QCoreApplication.translate("MainWindow", u"Average Results", None))
        self.modelFittingBtn_2.setText(QCoreApplication.translate("MainWindow", u"Intermediate Calculations", None))
        self.modelFittingBtn.setText(QCoreApplication.translate("MainWindow", u"Model Fitting", None))
        self.visualiseResultsBtn.setText(QCoreApplication.translate("MainWindow", u"Visualise Models", None))
    # retranslateUi

