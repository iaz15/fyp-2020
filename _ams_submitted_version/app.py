from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtWidgets
import sys
import os

import rename_files
import data_processing

import averaging_data

import optimisation_lubricant_eval
import test_file

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.ui1 = Ui_ManagerWidget()
        self.initUI1()

    ### Functions belonging to ui1 ###
    def initUI1(self):
        # Set up window 1
        self.ui1.setupUi(self)

        self.ui1.fileRenaming_btn.clicked.connect(self.launch_file_renaming)
        self.ui1.processData_btn.clicked.connect(self.launch_process_data)
        self.ui1.launchAverageDataWindow_btn.clicked.connect(self.launch_average_data_window)
        self.ui1.launchFitDataWindow_btn.clicked.connect(self.launch_fit_data_window)

        self.ui1.launchPlottingWindow_btn.clicked.connect(self.launch_second_window)

    def launch_file_renaming(self):
        print("Launching file renaming")
        rename_files.main()

    def launch_process_data(self):
        print("Launching data processing program")
        data_processing.main()

    def launch_average_data_window(self):
        print("Launching Averaging Data Window")
        self.initUI2()

    def launch_fit_data_window(self):
        print("Launching Fitting Data Window")
        self.initUI3()

    def launch_second_window(self):
        # Launch the new window
        self.initUI4()

    ### Functions belonging to ui2 (Averaging Data Window) ###
    def initUI2(self):
        self.AveragingDataWindow = QtWidgets.QWidget()
        self.ui2 = Ui_AveragingDataWidget()
        self.ui2.setupUi(self.AveragingDataWindow)

        self.ui2.averageData_btn.clicked.connect(self.launch_average_data)
        self.ui2.test_btn.clicked.connect(self.count_nums)

        self.AveragingDataWindow.show()

    def count_nums(self):
        test_file.count_to_num()

    def launch_average_data(self):
        print("Launching averaging program")
        (groupID, temperature, speed, force, volume) = self.get_test_conditions()

        return_val = averaging_data.gui_avg_datasets(group_id_chosen=groupID, temperature_chosen=temperature,
                                        speed_chosen=speed, force_chosen=force, volume_chosen=volume)

        if return_val == 0:
            # If it returns 0, that means it was not sucessful in averaging
            self.launch_popup('Invalid Test Conditions')

    def get_test_conditions(self):
        groupID = self.ui2.groupID_spinBox.value()
        temperature = self.ui2.temperature_SpinBox.value()
        speed = self.ui2.speed_spinBox.value()
        force = self.ui2.force_spinBox.value()
        volume = self.ui2.volume_spinBox.value()

        print(groupID, temperature, speed, force, volume)

        return (groupID, temperature, speed, force, volume)

    ### Functions belonging to ui3 (Fitting Window) ###
    def initUI3(self):
        self.FittingWindow = QtWidgets.QWidget()
        self.ui3 = Ui_FittingWidget()
        self.ui3.setupUi(self.FittingWindow)

        self.ui3.plotResults_btn.clicked.connect(self.launch_plot_results)
        self.ui3.manualFitting_btn.clicked.connect(self.launch_manual_fitting)
        self.ui3.automaticFitting_btn.clicked.connect(self.launch_automatic_fitting)

        self.FittingWindow.show()

    def get_group_id(self):
        groupID = int(self.ui3.groupID_spinBox.value())

        print(groupID)
        return groupID

    def launch_popup(self, i):
        self.message_box(i)
        print(i)

    def message_box(self, i):
        msg = QMessageBox()
        msg.setWindowTitle("Warning!")
        msg.setText(i)
        msg.setIcon(QMessageBox.Critical)
        x = msg.exec_()  # Shows the message box
        return(i)

    def launch_plot_results(self):
        print("Launching plot results program")
        groupID = self.get_group_id()
        return_val = optimisation_lubricant_eval.plot_results_rerun(groupID)

        if return_val == 0:
            # If it returns 0, that means it was not sucessful in averaging
            self.launch_popup('Invalid Group ID')

    def launch_manual_fitting(self):
        print("Launching manual fitting program")
        groupID = self.get_group_id()
        return_val = optimisation_lubricant_eval.manual_fitting_slider(groupID)

        if return_val == 0:
            # If it returns 0, that means it was not sucessful in averaging
            self.launch_popup('Invalid Group ID')

    def launch_automatic_fitting(self):
        print("Launching automatic fitting program")
        groupID = self.get_group_id()
        return_val = optimisation_lubricant_eval.optimisation_friction_model(groupID)

        if return_val == 0:
            # If it returns 0, that means it was not sucessful in averaging
            self.launch_popup('Invalid Group ID')
            
    ### Functions belonging to ui4 ###
    def initUI4(self):
        # Sets up window 2
        self.PlottingWindow = QtWidgets.QWidget()
        self.ui4 = Ui_Form()
        self.ui4.setupUi(self.PlottingWindow)

        self.ui4.plotWidget.canvas.fig.tight_layout()
        self.ui4.plotWidget.canvas.axes.grid()

        self.ui4.plotData_btn.clicked.connect(self.plot_data)
        self.ui4.clearGraph_btn.clicked.connect(self.clear_plot)
        self.ui4.processData_btn.clicked.connect(self.process_data)
        self.ui4.exit_btn.clicked.connect(self.exit_program)
        self.ui4.changeText_btn.clicked.connect(self.change_text)

        self.PlottingWindow.show()


    def plot_data(self):
        self.ui4.plotWidget.canvas.axes.cla()
        x = [1, 4, 6, 8, 9, 10]
        y = [5, 5, 2, 6, 11, 20]

        self.ui4.plotWidget.canvas.axes.plot(x, y)

        self.ui4.plotWidget.canvas.axes.set_title("Plot Number 1")
        self.ui4.plotWidget.canvas.axes.grid()
        self.ui4.plotWidget.canvas.axes.set_xlabel("X Axis")
        self.ui4.plotWidget.canvas.axes.set_ylabel("Y Axis")
        self.ui4.plotWidget.canvas.axes.set_xlim(0)
        self.ui4.plotWidget.canvas.axes.set_ylim(0)
        self.ui4.plotWidget.canvas.fig.tight_layout()
        self.ui4.plotWidget.canvas.draw()

    def clear_plot(self):
        self.ui4.plotWidget.canvas.axes.cla()
        self.ui4.plotWidget.canvas.axes.grid()
        self.ui4.plotWidget.canvas.draw()

    def process_data(self):
        optimisation_lubricant_eval.manual_fitting_slider()

    def change_text(self):
        self.ui4.label.setText("Second Window 1111")
        # When the text is changed, adjust the size of the label.
        self.ui4.label.adjustSize()

    def exit_program(self):
        print("Closing Program")
        sys.exit()




if __name__ == "__main__":
    # https://gist.github.com/MalloyDelacroix/2c509d6bcad35c7e35b1851dfc32d161
    # Generate the ui file
    # os.system("pyside2-uic manager_widget.ui -o ui_manager_widget.py")
    os.system("pyuic5 plotting_widget.ui -o ui_plotting_widget.py")
    os.system("pyuic5 manager_widget.ui -o ui_manager_widget.py")
    os.system("pyuic5 averaging_data_widget.ui -o ui_averaging_data_widget.py")
    os.system("pyuic5 fitting_widget.ui -o ui_fitting_widget.py")

    from ui_plotting_widget import Ui_Form
    from ui_manager_widget import Ui_ManagerWidget
    from ui_averaging_data_widget import Ui_AveragingDataWidget
    from ui_fitting_widget import Ui_FittingWidget

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    main_window = MainWindow()
    main_window.show()

    app.exec_()
