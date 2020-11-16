from PySide2.QtWidgets import QApplication, QMainWindow, QWidget
from PySide2.QtWidgets import QMessageBox
from PySide2 import QtWidgets
from pyqtgraph import PlotWidget, plot
from PySide2 import QtCore

import os
import shutil
from ui_designs.TestConditionsWindow import Ui_TestConditionsWindow
from ui_designs.MatplotlibWindow import Ui_MatplotlibWindow
from MatplotlibWindow import MatplotlibWindow_

import rename_files
import data_processing

class Ui_TestConditionsWindow_(QWidget, Ui_TestConditionsWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialise the extra plotting windows
        self.forceWindow = MatplotlibWindow_()
        self.speedWindow = MatplotlibWindow_()
        self.forceWindow.setWindowTitle('Force Plots')
        self.speedWindow.setWindowTitle('Speed Plots')

        self.plotWidget.canvas.fig.tight_layout()

        self.input_directory = 'friction_raw_data'
        self.results_directory = 'results_friction'
        self.bin_directory = 'friction_raw_data_bin'

        self.experiment_ids = []
        self.current_experiment_id_index = 0
        self.current_experiment_id = 0
        self.experiments = {}

        # Populate input boxes
        lubricant_ids = ['1', '2', '3', '4']
        self.lubricantIdComboBox.addItems(lubricant_ids)

        pin_materials = ['P20', 'H13']
        self.pinMaterialComboBox.addItems(pin_materials)

        blank_materials = ['AA6082', 'AA7075']
        self.blankMaterialComboBox.addItems(blank_materials)

        coating_materials = ['None', 'AlCrN', 'CrN']
        self.coatingMaterialComboBox.addItems(coating_materials)

        # Initialise on click events
        self.previousBtn.clicked.connect(self.previous_experiment)
        self.nextBtn.clicked.connect(self.next_experiment)
        self.checkFolderBtn.clicked.connect(self.check_folder_contents)
        self.testBtn.clicked.connect(self.test_btn_clicked)
        self.experimentsListWidget.itemClicked.connect(self.test_fcn)
        self.viewForcePlotBtn.clicked.connect(self.view_force_plot)
        self.viewSpeedPlotBtn.clicked.connect(self.view_position_plot)

        self.coatingMaterialComboBox.currentTextChanged.connect(self.update_coating_combobox)

        self.nextBtn.setEnabled(False)
        self.previousBtn.setEnabled(False)
    
    def displayUi(self):
        self.show()

    def view_force_plot(self):
        self.forceWindow.displayUi()
        self.forceWindow.plotWidget.canvas.fig.tight_layout()

    def view_position_plot(self):
        self.speedWindow.displayUi()
        self.speedWindow.plotWidget.canvas.fig.tight_layout()

    def test_fcn(self, list_item):
        print(list_item.text())

    def update_coating_combobox(self, text):
        if text == "None":
            self.coatingRoughnessDoubleSpinBox.setRange(0,0)
            self.coatingThicknessDoubleSpinBox.setRange(0,0)

            self.coatingRoughnessDoubleSpinBox.setValue(0.8)
            self.coatingThicknessDoubleSpinBox.setValue(0.9)
        else:
            self.coatingRoughnessDoubleSpinBox.setRange(0.1, 2)
            self.coatingThicknessDoubleSpinBox.setRange(0.1, 10)


    def plot_content(self):

        self.plotWidget.canvas.axes.cla()
        self.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['coefficient_of_friction'])
        self.plotWidget.canvas.axes.set_title('CoF vs SD')
        self.plotWidget.canvas.axes.set_xlabel('sliding distance (mm)')
        self.plotWidget.canvas.axes.set_ylabel('coefficient of friction')
        self.plotWidget.canvas.axes.set_ylabel('coefficient of friction')
        self.plotWidget.canvas.fig.tight_layout()
        self.plotWidget.canvas.draw()

        self.speedWindow.plotWidget.canvas.axes.cla()
        self.speedWindow.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['speed_x_(mm_s^-1)'], label='speed x (mm/s)')
        self.speedWindow.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['speed_y_(mm_s^-1)'], label='speed y (mm/s)')
        self.speedWindow.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['speed_z_(mm_s^-1)'], label='speed z (mm/s)')
        self.speedWindow.plotWidget.canvas.axes.set_xlabel('sliding distance (mm)')
        self.speedWindow.plotWidget.canvas.axes.set_ylabel('speed (mm/s)')
        self.speedWindow.plotWidget.canvas.axes.legend()
        self.speedWindow.plotWidget.canvas.draw()

        self.forceWindow.plotWidget.canvas.axes.cla()
        self.forceWindow.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['x_force_(N)'], label='force x (N)')
        self.forceWindow.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['y_force_(N)'], label='force y (N)')
        self.forceWindow.plotWidget.canvas.axes.plot(self.dfs[self.current_experiment_id]['sliding_distance_(mm)'], self.dfs[self.current_experiment_id]['z_force_(N)'], label='force z (N)')
        self.forceWindow.plotWidget.canvas.axes.set_xlabel('sliding distance (mm)')
        self.forceWindow.plotWidget.canvas.axes.set_ylabel('force (N)')
        self.forceWindow.plotWidget.canvas.axes.legend()
        self.forceWindow.plotWidget.canvas.draw()

    def check_folder_contents(self):
        # Reset to start at the first experiment
        self.experiment_ids = []
        self.current_experiment_id_index = 0
        self.current_experiment_id = 0
        self.experiments_dict = {}
        self.dfs = {}

        # First check the folder exists
        data_processing.check_folder_exists(self.input_directory)
        data_processing.check_folder_exists(self.bin_directory)
        data_processing.check_folder_exists(self.results_directory)

        # Get filenames in the input directory
        filenames = rename_files.get_filenames(self.input_directory)

        # Extract invalid and valid filenames
        invalid_filenames, valid_filenames = rename_files.validate_filenames(filenames)

        if not valid_filenames:
            self.nextBtn.setEnabled(False)
            self.previousBtn.setEnabled(False)
            pass
        
        else:
            # Get experimental ids, sorted in ascending order and save them
            flat_valid_filenames = [item for sublist in valid_filenames for item in sublist]
            self.experiment_ids = sorted(set([int(rename_files.extract_experiment_id(f)) for f in flat_valid_filenames]))

            self.current_experiment_id = self.experiment_ids[self.current_experiment_id_index]
            self.experimentsListWidget.clear()

            self.nextBtn.setEnabled(True)
            self.previousBtn.setEnabled(False)
            self.nextBtn.setText("Next")

            if self.current_experiment_id == max(self.experiment_ids):
                self.nextBtn.setText("Confirm Changes")

            # Convert valid filenames into a flat list to be compatible with reading & processing modules
            valid_filenames_flat = [item for sublist in valid_filenames for item in sublist]

            # Read csv files
            dfs_raw_data = data_processing.read_data(valid_filenames_flat, load_location_raw=self.input_directory)

            ####### Robotic Arm Specific Starts #######
            # Process raw data (position & force) separately
            dfs_position_force_filtered = data_processing.process_force_position(dfs_raw_data)

            # Remove the .csv extension from filenames for matching filenames to be compatible with cof processing module
            matching_filenames = [(item[0].rstrip('.csv'), item[1].rstrip('.csv')) for item in valid_filenames]

            # Combine the force and position data to obtain CoF plots (output key example: data_1)
            self.dfs = data_processing.extract_cof_results(matching_filenames, dfs_position_force_filtered)

            self.plot_content()

            self.update_title()

    def next_experiment(self):
        lubricant_id = str(self.lubricantIdComboBox.currentText())

        pin_material = str(self.pinMaterialComboBox.currentText())
        pin_roughness = str(self.pinRoughnessDoubleSpinBox.value())

        blank_material = str(self.blankMaterialComboBox.currentText())
        blank_roughness = str(self.blankRoughnessDoubleSpinBox.value())
        blank_thickness = str(self.blankThicknessDoubleSpinBox.value())

        coating_material = str(self.coatingMaterialComboBox.currentText())
        coating_roughness = str(self.coatingRoughnessDoubleSpinBox.value())
        coating_thickness = str(self.coatingThicknessDoubleSpinBox.value())

        temperature = str(self.temperatureDoubleSpinBox.value())
        speed = str(self.speedDoubleSpinBox.value())
        force = str(self.forceDoubleSpinBox.value())
        pressure = str(self.pressureDoubleSpinBox.value())
        lubricant_thickness = str(self.lubricantThicknessDoubleSpinBox.value())

        experiment = rename_files.Experiment(self.current_experiment_id, lubricant_id, pin_material, pin_roughness,
                                             blank_material, blank_roughness, blank_thickness,
                                             coating_material, coating_roughness, coating_thickness,
                                             temperature, speed, force, pressure, lubricant_thickness)

        if self.current_experiment_id_index == len(self.experiment_ids) - 2:
            # If the current experiment is the largest one
            self.nextBtn.setText("Confirm Changes")

        if self.current_experiment_id_index < len(self.experiment_ids) - 1:
            # If it still hasn't reached the final experiment
            self.previousBtn.setEnabled(True)
            self.nextBtn.setEnabled(True)
            self.current_experiment_id_index += 1

            # If list is empty
            if not self.experimentsListWidget.findItems(str(self.current_experiment_id), QtCore.Qt.MatchExactly):
                self.experimentsListWidget.addItem(str(self.current_experiment_id))

            self.experiments_dict[experiment.id] = experiment
            self.current_experiment_id = self.experiment_ids[self.current_experiment_id_index]

            self.plot_content()

            self.update_title()

        elif self.current_experiment_id_index == len(self.experiment_ids) - 1:
            # If the current index is the final index, all results have been processed
            self.nextBtn.setEnabled(False)
            self.previousBtn.setEnabled(False)

            self.experiments_dict[experiment.id] = experiment
            self.current_experiment_id = self.experiment_ids[self.current_experiment_id_index]

            # If list is empty
            if not self.experimentsListWidget.findItems(str(self.current_experiment_id), QtCore.Qt.MatchExactly):
                self.experimentsListWidget.addItem(str(self.current_experiment_id))

            # Save the results in the dataframes stored into files
            # For each processed experiment
            for idx, (exp_num, value) in enumerate(self.dfs.items(), 1):

                lubricant_id = self.experiments_dict[exp_num].conditions.lubricant_id
                blank_material = self.experiments_dict[exp_num].conditions.blank_material

                output_filename = f"{exp_num}_L{lubricant_id}_{blank_material}.csv"

                # Save the filenames in the corresponding experiment object
                self.experiments_dict[exp_num].add_output_filename(output_filename)

                # Save the output files
                output_filepath = os.path.join(self.results_directory, output_filename)
                self.dfs[exp_num].to_csv(output_filepath, sep=',', index=False)
                print(f"saved: {exp_num}")

                # Move the processed data to the bin folder
                input_filename_force = f"force_data_{exp_num}.csv"
                input_filename_position = f"position_data_{exp_num}.csv"

                input_path_force = os.path.join(self.input_directory, input_filename_force)
                input_path_position = os.path.join(self.input_directory, input_filename_position)

                bin_path_force = os.path.join(self.bin_directory, input_filename_force)
                bin_path_position = os.path.join(self.bin_directory, input_filename_position)
                
                shutil.move(input_path_force, bin_path_force)
                shutil.move(input_path_position, bin_path_position)

    def previous_experiment(self):

        if self.current_experiment_id_index != 0:
            self.nextBtn.setText("Next")

        self.current_experiment_id_index -= 1
        self.current_experiment_id = self.experiment_ids[self.current_experiment_id_index]

        if self.current_experiment_id_index == 0:
            self.previousBtn.setEnabled(False)
        else:
            self.previousBtn.setEnabled(True)

        self.plot_content()

        self.nextBtn.setEnabled(True)
        self.update_title()

    def test_btn_clicked(self):
        print("Test button clicked!")
        print(self.experiments)

    def update_title(self):
        self.titleLabel.setText(f"Experiment ID: {self.current_experiment_id} ({self.current_experiment_id_index+1}/{len(self.experiment_ids)})")
