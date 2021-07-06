import pandas as pd
import os
import glob
import numpy as np
import re

# robot files -> file with experimental conditions (will be used later in the system)

# extract the fixed values from the file name (experiment id & file type)


def get_filename_metadata(filename):
    return

# populate the parameters for each of the experiments
#   but if files have the same id, then they will share the same test conditions

# how the user interacts with the system
# determine the present experimental IDs with force & position data (this should always come in pairs)
#   warn the user if there are orphaned files (let them know what the file names are) and don't include in processing
# for each experimental ID, the user will input the experimental conditions

# user fills in form
# user clicks save -> initialise the class
class Experiment:
    def __init__(self, experiment_id, lubricant, pin_material, pin_roughness_Ra, blank_material, blank_roughness_Ra, blank_thickness_mm,
                 coating_material, coating_thickness_mm, coating_roughness_Ra,
                 temperature_degC, speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers):
        self.id = experiment_id

        self.conditions = ExperimentalConditions(lubricant, pin_material, pin_roughness_Ra, blank_material, blank_roughness_Ra, blank_thickness_mm,
                                                 coating_material, coating_thickness_mm, coating_roughness_Ra,
                                                 temperature_degC, speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers)

    def add_output_filename(self, filename):
        self.filename = filename

class ExperimentalConditions:
    def __init__(self, lubricant, pin_material, pin_roughness_Ra, blank_material, blank_roughness_Ra, blank_thickness_mm,
                 coating_material, coating_thickness_mm, coating_roughness_Ra,
                 temperature_degC, speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers, equiv_solid_thickness_micrometers=None):

        self.lubricant = lubricant

        self.pin_material = pin_material
        self.pin_roughness_Ra = pin_roughness_Ra

        self.blank_material = blank_material
        self.blank_roughness_Ra = blank_roughness_Ra
        self.blank_thickness_mm = blank_thickness_mm

        self.coating_material = coating_material
        self.coating_thickness_mm = coating_thickness_mm
        self.coating_roughness_Ra = coating_roughness_Ra

        self.temperature_degC = temperature_degC
        self.force_N = force_N
        self.pressure_MPa = pressure_MPa
        self.speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers = speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers
        self.lubricant_thickness_micrometers = lubricant_thickness_micrometers
        self.equiv_solid_thickness_micrometers = equiv_solid_thickness_micrometers

def get_filenames(directory):
    return os.listdir(directory)


def validate_filenames(filenames):
    '''
    returns two lists of filenames,
    first item is all invalid filenames and files without a pair
    sedond item is all paired valid filenames
    '''
    filename_pattern = '(force|position)_data_([0-9]+).csv'
    filename_matcher = re.compile(filename_pattern)

    matched_filenames = [f for f in filenames if filename_matcher.match(f)]
    invalid_filenames = [f for f in filenames if not filename_matcher.match(f)]

    # even if a file matches the pattern, we only care about pairs
    paired_filenames = []
    orphaned_filenames = []

    for f in matched_filenames:

        if 'force' in f:
            position_pair = f.replace('force', 'position')
            if position_pair in matched_filenames:
                position_filename = matched_filenames.pop(matched_filenames.index(position_pair))

                # Append paired file names to a list in tuples
                paired_filenames.append((position_filename, f))
            else:
                orphaned_filenames.append(f)

        elif 'position' in f:
            force_pair = f.replace('position', 'force')
            if force_pair in matched_filenames:
                force_filename = matched_filenames.pop(matched_filenames.index(force_pair))

                # Append paired file names to a list in tuples
                paired_filenames.append((force_filename, f))
            else:
                orphaned_filenames.append(f)

    invalid_filenames.extend(orphaned_filenames)

    return invalid_filenames, paired_filenames

def create_experiment(experiment_id, lubricant, pin_material, pin_roughness_Ra, blank_material, blank_roughness_Ra, blank_thickness_mm,
                                coating_material, coating_roughness_Ra, coating_thickness_mm,
                                temperature_degC, speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers):

    experiment = Experiment(experiment_id, lubricant, pin_material, pin_roughness_Ra, blank_material, blank_roughness_Ra, blank_thickness_mm,
                                coating_material, coating_roughness_Ra, coating_thickness_mm,
                                temperature_degC, speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers)
    return experiment

# todo: placeholder for ui event handling
def populate_experiment(experiment_id=1):
    # user will pass in:
    # lubricant, pin_material, pin_roughness_Ra, blank_material, blank_roughness_Ra,
    # temperature_degC, force_N, speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers, lubricant_thickness_micrometers

    # this information will be stored in a list
    experiment_ids = [7, 9]

    lubricants = ['Omega 35', 'Omega 35']
    pin_material = ['P20', 'P20']
    pin_roughness_Ra = [0.4, 0.4]
    blank_material = ['AA7075', 'AA7075']
    blank_roughness_Ra = [0.9, 0.9]

    blank_thickness_mm = [2, 2]
    coating_material = ['AA7075', 'AA7075']
    coating_thickness_mm = [0.8, 0.8]
    coating_roughness_Ra = [0.7, 0.7]

    temperature_degC = [400, 450]
    force_N = [8, 8]
    speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers = [100, 100]
    lubricant_thickness_micrometers = [10, 10]

    if experiment_id==1:
        experiment = Experiment(experiment_ids[0], lubricants[0], pin_material[0], pin_roughness_Ra[0], blank_material[0], blank_roughness_Ra[0], blank_thickness_mm[0],
        coating_material[0], coating_thickness_mm[0], coating_roughness_Ra[0],
        temperature_degC[0], force_N[0], speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers[0], lubricant_thickness_micrometers[0])

    elif experiment_id==2:
        experiment = Experiment(experiment_ids[1], lubricants[1], pin_material[1], pin_roughness_Ra[1], blank_material[1], blank_roughness_Ra[1], blank_thickness_mm[1],
        coating_material[1], coating_thickness_mm[1], coating_roughness_Ra[1],
        temperature_degC[1], force_N[1], speed_mmpersecond, force_N, pressure_MPa, lubricant_thickness_micrometers[1], lubricant_thickness_micrometers[1])

    else:
        return None

    return experiment

def rename_files_(directory, experiments):
    """
    Takes in directory with files to be renamed as well as experiments
    which is a dictionary containing Experiment objects
    """

    for key, exp_object in experiments.items():
        print(key)
        print(exp_object.conditions.temperature_degC)
        print(exp_object.conditions.coating_thickness_mm)

    return None

def extract_experiment_id(filename):
    """
    Given a valid filename, return the corresponding experiment id.
    """
    filename_pattern = '(force|position)_data_([0-9]+).csv'
    filename_matcher = re.compile(filename_pattern)

    exp_num = filename_matcher.match(filename).group(2)
    return exp_num

# this is our main block
def main():
    # run this block first when opening the application
    input_directory = 'friction_raw_data'
    filenames = get_filenames(input_directory)

    invalid_filenames, valid_filenames = validate_filenames(filenames)

    # get experimental ids, sorted in ascending order
    exp_ids = sorted(set([int(extract_experiment_id(f)) for f in valid_filenames]))
    print(f'Experimental IDs: {exp_ids}')

    # UI loop
    # show all experiment numbers as a list
    # user selects an item and it will show a form to allow inputting the test conditions

    # get experimental test conditions from the user for each experiment ID
    experiments = {}
    experiment = populate_experiment(experiment_id=1) # pass in fields from ui
    experiments[experiment.id] = experiment
    experiment = populate_experiment(experiment_id=2) # pass in fields from ui
    experiments[experiment.id] = experiment

    print(experiments)
    # user confirms all changes
    # validate to make sure all of the experimental files have been initialised
    #   if not valid, show error dialog and dont continue
    # otherwise, do the action and show success dialog
    rename_files_(input_directory, experiments)
    # close window and show main screen


def create_filename_string(exp_num, data_type, test_params, extension):
    """
    exp_num is the experiment number (an integer).

    data_type is a string either with "position" or "force".

    test_params is a dictionary with test conditions (e.g. F, v, etc).

    extension is a string (e.g. .csv).
    """
    keys = ["lubricant", "pinMat", "pinRa", "blankMat", "blankRa", "T", "F", "v", "V"]
    # 15_position_lubricant=1_pinMat=P20_pinRa=0.8_blankMat=AA7075_blankRa=0.3_T=300_F=6.5_v=50_V=23.2.csv
    # {experiment_id}_{file_type(force or position)}_{parameter=value}_*.csv
    # all parmameters must be present

    filename_parts = [str(exp_num), data_type]
    for key in keys:
        test_condition = str(key) + "=" + str(test_params[key])
        filename_parts.append(test_condition)

    filename = "_".join(filename_parts) + ".csv"
    return filename

if __name__ == "__main__":
    main()