import pandas as pd
import os
import glob
import numpy as np
import re


def create_filename_string(exp_num, data_type, test_params, extension):
    """
    exp_num is the experiment number (an integer).

    data_type is a string either with "position" or "force".

    test_params is a dictionary with test conditions (e.g. F, v, etc).

    extension is a string (e.g. .csv).
    """
    keys = ["lubricantid", "pinMat", "pinRa", "blankMat", "blankRa", "T", "F", "v", "V"]
    # 15_position_lubricantid=1_pinMat=P20_pinRa=0.8_blankMat=AA7075_blankRa=0.3_T=300_F=6.5_v=50_V=23.2.csv

    filename_parts = [str(exp_num), data_type]
    for key in keys:
        test_condition = str(key) + "=" + str(test_params[key])
        filename_parts.append(test_condition)

    filename = "_".join(filename_parts) + ".csv"
    return filename


def main():
    # Assume room temperature is 20 Degrees Celcius

    # We have force_data_7.csv etc and position_data_7.csv
    # Desired filename example:
    # 15_position_lubricantid=1_pinMat=P20_pinRa=0.8_blankMat=AA7075_blankRa=0.3_T=300_F=6.5_v=50_V=23.2.csv

    # Total of 9 properties to set
    # Possible values
    temperatures = [20, 200, 250, 300, 350, 400]
    speed = [10, 30, 50, 80, 100]
    load = [5, 6.5, 8]
    volume = [23.2]

    # Fixed values
    lubricantid = [1]
    pinMat = ['P20']
    pinRa = [0.8]
    blankMat = ['AA7075']
    blankRa = [0.3]

    # Fixed values will be set in the dictionary...
    # At the moment temperature is fixed at 20 and volume at 23.2
    test_parameters_base = dict(lubricantid=1, pinMat='P20', pinRa=0.8, blankMat='AA7075', blankRa=0.3)
    temp_fixed_params = dict(T=20, V=23.2)

    test_parameters_base.update(temp_fixed_params)

    # The user will give the file numbers and then the other test
    # conditions that are changing (T, v, F, V for example) for each test number.

    extension = '.csv'

    set_exp_nums = set()
    file_directory = "friction_raw_data"
    for file in os.listdir(file_directory):
        if file.endswith(extension):
            regex_ints =  re.compile("[0-9]+")
            exp_num = regex_ints.findall(file)
            # It ouputs a list, but there should only be one match.
            set_exp_nums.add(exp_num[0])

    ########## USER INPUT ##########
    # Add the filenames here.
    # Note 26 - 28 are with lubricant (The others are not)
    files_8N = [7, 9, 10, 11, 12, 13, 14, 15]
    files_6_5N = [16, 17, 18, 19, 20]
    files_5N = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44]

    files_50mm_s = [7, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 40, 41, 42, 43, 44]
    files_80mm_s = [13, 14, 15]
    files_30mm_s = [29, 30, 31, 32, 33]
    files_10mm_s = [34, 35, 36, 38, 39]

    force_values = {'8': files_8N}
    force_values = {'8': files_8N, '6.5': files_6_5N, '5': files_5N}
    speed_values = {'50': files_50mm_s}
    speed_values = {'50': files_50mm_s, '80': files_80mm_s, '30': files_30mm_s, '10': files_10mm_s}
    ########## USER INPUT ##########

    exp_to_force = {}
    exp_to_speed = {}
    for key, exp_list in force_values.items():
        for exp in exp_list:
            exp_to_force[exp] = key

    for key, exp_list in speed_values.items():
        for exp in exp_list:
            exp_to_speed[exp] = key

    # Dictionary containing experiment numbers matched to dictionaries with the test conditions.
    map_exp_conditions = {}
    for exp_num in set_exp_nums:
        # First add a copy of the base dictionary
        map_exp_conditions[int(exp_num)] = test_parameters_base.copy()

        # First add the force
        map_exp_conditions[int(exp_num)]['F'] = exp_to_force[int(exp_num)]

        # Then add the speed
        map_exp_conditions[int(exp_num)]['v'] = exp_to_speed[int(exp_num)]

    # Temperature and Volume are fixed at 20 and 23.2 respectively

    # Then loop again for renaming
    count = 0
    for file in os.listdir(file_directory):
        if file.endswith(extension):
            regex_ints =  re.compile("[0-9]+")
            exp_num = int(regex_ints.findall(file)[0])
            if "force" in file:
                dtype = "force"
                new_filename = create_filename_string(exp_num, dtype, map_exp_conditions[exp_num], extension)
                os.rename(os.path.join(file_directory, file), os.path.join(file_directory, new_filename))
                count += 1

            elif "position" in file:
                dtype = "position"
                new_filename = create_filename_string(exp_num, dtype, map_exp_conditions[exp_num], extension)
                os.rename(os.path.join(file_directory, file), os.path.join(file_directory, new_filename))
                count += 1

            # print(f"Renaming: {file}")
            # print(f"New name: {new_filename}")

    print(f"Renamed {count} files")


if __name__ == "__main__":
    main()
