# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'TestConditionsWindow.ui'
##
## Created by: Qt User Interface Compiler version 5.15.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from mplwidget import MplWidget


class Ui_TestConditionsWindow(object):
    def setupUi(self, TestConditionsWindow):
        if not TestConditionsWindow.objectName():
            TestConditionsWindow.setObjectName(u"TestConditionsWindow")
        TestConditionsWindow.resize(896, 648)
        TestConditionsWindow.setStyleSheet(u"")
        self.gridLayout = QGridLayout(TestConditionsWindow)
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.testConditionsSet1 = QGroupBox(TestConditionsWindow)
        self.testConditionsSet1.setObjectName(u"testConditionsSet1")
        font = QFont()
        font.setPointSize(9)
        self.testConditionsSet1.setFont(font)
        self.verticalLayout = QVBoxLayout(self.testConditionsSet1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.lubricantIdLabel = QLabel(self.testConditionsSet1)
        self.lubricantIdLabel.setObjectName(u"lubricantIdLabel")
        self.lubricantIdLabel.setFont(font)

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.lubricantIdLabel)

        self.lubricantIdComboBox = QComboBox(self.testConditionsSet1)
        self.lubricantIdComboBox.setObjectName(u"lubricantIdComboBox")
        self.lubricantIdComboBox.setFont(font)

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.lubricantIdComboBox)

        self.pinMaterialLabel = QLabel(self.testConditionsSet1)
        self.pinMaterialLabel.setObjectName(u"pinMaterialLabel")
        self.pinMaterialLabel.setFont(font)

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.pinMaterialLabel)

        self.pinMaterialComboBox = QComboBox(self.testConditionsSet1)
        self.pinMaterialComboBox.setObjectName(u"pinMaterialComboBox")
        self.pinMaterialComboBox.setFont(font)

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.pinMaterialComboBox)

        self.pinRoughnessLabel = QLabel(self.testConditionsSet1)
        self.pinRoughnessLabel.setObjectName(u"pinRoughnessLabel")
        self.pinRoughnessLabel.setFont(font)

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.pinRoughnessLabel)

        self.pinRoughnessDoubleSpinBox = QDoubleSpinBox(self.testConditionsSet1)
        self.pinRoughnessDoubleSpinBox.setObjectName(u"pinRoughnessDoubleSpinBox")
        self.pinRoughnessDoubleSpinBox.setFont(font)
        self.pinRoughnessDoubleSpinBox.setDecimals(1)
        self.pinRoughnessDoubleSpinBox.setMinimum(0.100000000000000)
        self.pinRoughnessDoubleSpinBox.setMaximum(2.000000000000000)
        self.pinRoughnessDoubleSpinBox.setSingleStep(0.100000000000000)

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.pinRoughnessDoubleSpinBox)

        self.blankMaterialLabel = QLabel(self.testConditionsSet1)
        self.blankMaterialLabel.setObjectName(u"blankMaterialLabel")
        self.blankMaterialLabel.setFont(font)

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.blankMaterialLabel)

        self.blankMaterialComboBox = QComboBox(self.testConditionsSet1)
        self.blankMaterialComboBox.setObjectName(u"blankMaterialComboBox")
        self.blankMaterialComboBox.setFont(font)

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.blankMaterialComboBox)

        self.blankRoughnessLabel = QLabel(self.testConditionsSet1)
        self.blankRoughnessLabel.setObjectName(u"blankRoughnessLabel")
        self.blankRoughnessLabel.setFont(font)

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.blankRoughnessLabel)

        self.blankRoughnessDoubleSpinBox = QDoubleSpinBox(self.testConditionsSet1)
        self.blankRoughnessDoubleSpinBox.setObjectName(u"blankRoughnessDoubleSpinBox")
        self.blankRoughnessDoubleSpinBox.setFont(font)
        self.blankRoughnessDoubleSpinBox.setDecimals(1)
        self.blankRoughnessDoubleSpinBox.setMinimum(0.100000000000000)
        self.blankRoughnessDoubleSpinBox.setMaximum(2.000000000000000)
        self.blankRoughnessDoubleSpinBox.setSingleStep(0.100000000000000)

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.blankRoughnessDoubleSpinBox)

        self.blankRoughnessLabel_2 = QLabel(self.testConditionsSet1)
        self.blankRoughnessLabel_2.setObjectName(u"blankRoughnessLabel_2")
        self.blankRoughnessLabel_2.setFont(font)

        self.formLayout_3.setWidget(5, QFormLayout.LabelRole, self.blankRoughnessLabel_2)

        self.blankThicknessDoubleSpinBox = QDoubleSpinBox(self.testConditionsSet1)
        self.blankThicknessDoubleSpinBox.setObjectName(u"blankThicknessDoubleSpinBox")
        self.blankThicknessDoubleSpinBox.setFont(font)
        self.blankThicknessDoubleSpinBox.setDecimals(1)
        self.blankThicknessDoubleSpinBox.setMinimum(0.100000000000000)
        self.blankThicknessDoubleSpinBox.setMaximum(10.000000000000000)
        self.blankThicknessDoubleSpinBox.setSingleStep(0.100000000000000)

        self.formLayout_3.setWidget(5, QFormLayout.FieldRole, self.blankThicknessDoubleSpinBox)

        self.coatingMaterialLabel = QLabel(self.testConditionsSet1)
        self.coatingMaterialLabel.setObjectName(u"coatingMaterialLabel")
        self.coatingMaterialLabel.setFont(font)

        self.formLayout_3.setWidget(6, QFormLayout.LabelRole, self.coatingMaterialLabel)

        self.coatingMaterialComboBox = QComboBox(self.testConditionsSet1)
        self.coatingMaterialComboBox.setObjectName(u"coatingMaterialComboBox")
        self.coatingMaterialComboBox.setFont(font)

        self.formLayout_3.setWidget(6, QFormLayout.FieldRole, self.coatingMaterialComboBox)

        self.coatingThicknessLabel_2 = QLabel(self.testConditionsSet1)
        self.coatingThicknessLabel_2.setObjectName(u"coatingThicknessLabel_2")
        self.coatingThicknessLabel_2.setFont(font)

        self.formLayout_3.setWidget(7, QFormLayout.LabelRole, self.coatingThicknessLabel_2)

        self.coatingRoughnessDoubleSpinBox = QDoubleSpinBox(self.testConditionsSet1)
        self.coatingRoughnessDoubleSpinBox.setObjectName(u"coatingRoughnessDoubleSpinBox")
        self.coatingRoughnessDoubleSpinBox.setFont(font)
        self.coatingRoughnessDoubleSpinBox.setDecimals(1)
        self.coatingRoughnessDoubleSpinBox.setMinimum(0.000000000000000)
        self.coatingRoughnessDoubleSpinBox.setMaximum(0.000000000000000)
        self.coatingRoughnessDoubleSpinBox.setSingleStep(0.100000000000000)

        self.formLayout_3.setWidget(7, QFormLayout.FieldRole, self.coatingRoughnessDoubleSpinBox)

        self.coatingThicknessLabel = QLabel(self.testConditionsSet1)
        self.coatingThicknessLabel.setObjectName(u"coatingThicknessLabel")
        self.coatingThicknessLabel.setFont(font)

        self.formLayout_3.setWidget(8, QFormLayout.LabelRole, self.coatingThicknessLabel)

        self.coatingThicknessDoubleSpinBox = QDoubleSpinBox(self.testConditionsSet1)
        self.coatingThicknessDoubleSpinBox.setObjectName(u"coatingThicknessDoubleSpinBox")
        self.coatingThicknessDoubleSpinBox.setFont(font)
        self.coatingThicknessDoubleSpinBox.setDecimals(1)
        self.coatingThicknessDoubleSpinBox.setMinimum(0.000000000000000)
        self.coatingThicknessDoubleSpinBox.setMaximum(0.000000000000000)
        self.coatingThicknessDoubleSpinBox.setSingleStep(0.100000000000000)

        self.formLayout_3.setWidget(8, QFormLayout.FieldRole, self.coatingThicknessDoubleSpinBox)


        self.verticalLayout.addLayout(self.formLayout_3)


        self.verticalLayout_3.addWidget(self.testConditionsSet1)

        self.groupBox_2 = QGroupBox(TestConditionsWindow)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setInputMethodHints(Qt.ImhNone)
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.temperatureLabel = QLabel(self.groupBox_2)
        self.temperatureLabel.setObjectName(u"temperatureLabel")
        self.temperatureLabel.setFont(font)

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.temperatureLabel)

        self.temperatureDoubleSpinBox = QDoubleSpinBox(self.groupBox_2)
        self.temperatureDoubleSpinBox.setObjectName(u"temperatureDoubleSpinBox")
        self.temperatureDoubleSpinBox.setFont(font)
        self.temperatureDoubleSpinBox.setDecimals(1)
        self.temperatureDoubleSpinBox.setMinimum(0.000000000000000)
        self.temperatureDoubleSpinBox.setMaximum(1000.000000000000000)
        self.temperatureDoubleSpinBox.setSingleStep(10.000000000000000)

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.temperatureDoubleSpinBox)

        self.speedDoubleSpinBox_2 = QDoubleSpinBox(self.groupBox_2)
        self.speedDoubleSpinBox_2.setObjectName(u"speedDoubleSpinBox_2")
        self.speedDoubleSpinBox_2.setFont(font)
        self.speedDoubleSpinBox_2.setDecimals(1)
        self.speedDoubleSpinBox_2.setMinimum(1.000000000000000)
        self.speedDoubleSpinBox_2.setMaximum(500.000000000000000)
        self.speedDoubleSpinBox_2.setSingleStep(5.000000000000000)
        self.speedDoubleSpinBox_2.setValue(5.000000000000000)

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.speedDoubleSpinBox_2)

        self.forceLabel = QLabel(self.groupBox_2)
        self.forceLabel.setObjectName(u"forceLabel")
        self.forceLabel.setFont(font)

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.forceLabel)

        self.forceDoubleSpinBox = QDoubleSpinBox(self.groupBox_2)
        self.forceDoubleSpinBox.setObjectName(u"forceDoubleSpinBox")
        self.forceDoubleSpinBox.setFont(font)
        self.forceDoubleSpinBox.setDecimals(1)
        self.forceDoubleSpinBox.setMinimum(0.100000000000000)
        self.forceDoubleSpinBox.setSingleStep(0.500000000000000)
        self.forceDoubleSpinBox.setValue(0.500000000000000)

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.forceDoubleSpinBox)

        self.lubricantVolumeLabel = QLabel(self.groupBox_2)
        self.lubricantVolumeLabel.setObjectName(u"lubricantVolumeLabel")
        self.lubricantVolumeLabel.setFont(font)

        self.formLayout_2.setWidget(4, QFormLayout.LabelRole, self.lubricantVolumeLabel)

        self.lubricantThicknessDoubleSpinBox = QDoubleSpinBox(self.groupBox_2)
        self.lubricantThicknessDoubleSpinBox.setObjectName(u"lubricantThicknessDoubleSpinBox")
        self.lubricantThicknessDoubleSpinBox.setFont(font)
        self.lubricantThicknessDoubleSpinBox.setDecimals(1)
        self.lubricantThicknessDoubleSpinBox.setMinimum(0.100000000000000)
        self.lubricantThicknessDoubleSpinBox.setValue(1.000000000000000)

        self.formLayout_2.setWidget(4, QFormLayout.FieldRole, self.lubricantThicknessDoubleSpinBox)

        self.pressureDoubleSpinBox_2 = QDoubleSpinBox(self.groupBox_2)
        self.pressureDoubleSpinBox_2.setObjectName(u"pressureDoubleSpinBox_2")
        self.pressureDoubleSpinBox_2.setFont(font)
        self.pressureDoubleSpinBox_2.setDecimals(1)
        self.pressureDoubleSpinBox_2.setMinimum(0.100000000000000)
        self.pressureDoubleSpinBox_2.setSingleStep(0.500000000000000)
        self.pressureDoubleSpinBox_2.setValue(0.500000000000000)

        self.formLayout_2.setWidget(3, QFormLayout.FieldRole, self.pressureDoubleSpinBox_2)

        self.pressureLabel = QLabel(self.groupBox_2)
        self.pressureLabel.setObjectName(u"pressureLabel")
        self.pressureLabel.setFont(font)

        self.formLayout_2.setWidget(3, QFormLayout.LabelRole, self.pressureLabel)

        self.speedLabel = QLabel(self.groupBox_2)
        self.speedLabel.setObjectName(u"speedLabel")
        self.speedLabel.setFont(font)

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.speedLabel)


        self.verticalLayout_2.addLayout(self.formLayout_2)


        self.verticalLayout_3.addWidget(self.groupBox_2)


        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 2)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.previousBtn = QPushButton(TestConditionsWindow)
        self.previousBtn.setObjectName(u"previousBtn")
        self.previousBtn.setFont(font)
        self.previousBtn.setCheckable(False)

        self.horizontalLayout.addWidget(self.previousBtn)

        self.nextBtn = QPushButton(TestConditionsWindow)
        self.nextBtn.setObjectName(u"nextBtn")
        self.nextBtn.setFont(font)

        self.horizontalLayout.addWidget(self.nextBtn)


        self.verticalLayout_4.addLayout(self.horizontalLayout)

        self.checkFolderBtn = QPushButton(TestConditionsWindow)
        self.checkFolderBtn.setObjectName(u"checkFolderBtn")
        self.checkFolderBtn.setFont(font)

        self.verticalLayout_4.addWidget(self.checkFolderBtn)

        self.testBtn = QPushButton(TestConditionsWindow)
        self.testBtn.setObjectName(u"testBtn")
        self.testBtn.setFont(font)

        self.verticalLayout_4.addWidget(self.testBtn)


        self.gridLayout.addLayout(self.verticalLayout_4, 1, 0, 3, 1)

        self.experimentsListWidget = QListWidget(TestConditionsWindow)
        self.experimentsListWidget.setObjectName(u"experimentsListWidget")

        self.gridLayout.addWidget(self.experimentsListWidget, 1, 1, 3, 1)

        self.titleLabel = QLabel(TestConditionsWindow)
        self.titleLabel.setObjectName(u"titleLabel")
        font1 = QFont()
        font1.setPointSize(10)
        self.titleLabel.setFont(font1)
        self.titleLabel.setInputMethodHints(Qt.ImhNone)
        self.titleLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.titleLabel, 1, 2, 1, 1)

        self.viewForcePlotBtn = QPushButton(TestConditionsWindow)
        self.viewForcePlotBtn.setObjectName(u"viewForcePlotBtn")
        self.viewForcePlotBtn.setFont(font)

        self.gridLayout.addWidget(self.viewForcePlotBtn, 2, 2, 1, 1)

        self.viewSpeedPlotBtn = QPushButton(TestConditionsWindow)
        self.viewSpeedPlotBtn.setObjectName(u"viewSpeedPlotBtn")
        self.viewSpeedPlotBtn.setFont(font)
        self.viewSpeedPlotBtn.setStyleSheet(u"")

        self.gridLayout.addWidget(self.viewSpeedPlotBtn, 3, 2, 1, 1)

        self.groupBox = QGroupBox(TestConditionsWindow)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setFont(font)
        self.verticalLayout_5 = QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.plotWidget = MplWidget(self.groupBox)
        self.plotWidget.setObjectName(u"plotWidget")
        self.plotWidget.setMinimumSize(QSize(550, 420))
        self.plotWidget.setAutoFillBackground(False)

        self.verticalLayout_5.addWidget(self.plotWidget)


        self.gridLayout.addWidget(self.groupBox, 0, 2, 1, 1)


        self.retranslateUi(TestConditionsWindow)

        QMetaObject.connectSlotsByName(TestConditionsWindow)
    # setupUi

    def retranslateUi(self, TestConditionsWindow):
        TestConditionsWindow.setWindowTitle(QCoreApplication.translate("TestConditionsWindow", u"Form", None))
        self.testConditionsSet1.setTitle(QCoreApplication.translate("TestConditionsWindow", u"Test Conditions Set 1", None))
        self.lubricantIdLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Lubricant ID", None))
        self.pinMaterialLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Pin Material", None))
        self.pinRoughnessLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Pin Roughness (Ra)", None))
        self.blankMaterialLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Blank Material", None))
        self.blankRoughnessLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Blank Roughness (Ra)", None))
        self.blankRoughnessLabel_2.setText(QCoreApplication.translate("TestConditionsWindow", u"Blank Thickness (mm)", None))
        self.coatingMaterialLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Coating Material", None))
        self.coatingThicknessLabel_2.setText(QCoreApplication.translate("TestConditionsWindow", u"Coating Roughness (Ra)", None))
        self.coatingThicknessLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Coating Thickness (mm)", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("TestConditionsWindow", u"Test Conditions Set 2", None))
        self.temperatureLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Temperature (\u00b0C)", None))
        self.forceLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Force (N)", None))
        self.lubricantVolumeLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Lubricant Thickness (\u03bcm)", None))
        self.pressureLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Pressure (MPa)", None))
        self.speedLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Speed (mm/s)", None))
        self.previousBtn.setText(QCoreApplication.translate("TestConditionsWindow", u"Previous", None))
        self.nextBtn.setText(QCoreApplication.translate("TestConditionsWindow", u"Next", None))
        self.checkFolderBtn.setText(QCoreApplication.translate("TestConditionsWindow", u"Check Folder Contents", None))
        self.testBtn.setText(QCoreApplication.translate("TestConditionsWindow", u"Test Button", None))
        self.titleLabel.setText(QCoreApplication.translate("TestConditionsWindow", u"Experiment ID: None", None))
        self.viewForcePlotBtn.setText(QCoreApplication.translate("TestConditionsWindow", u"Show Force Plots", None))
        self.viewSpeedPlotBtn.setText(QCoreApplication.translate("TestConditionsWindow", u"Show Position/Speed Plots", None))
        self.groupBox.setTitle(QCoreApplication.translate("TestConditionsWindow", u"Plotting Result", None))
    # retranslateUi

