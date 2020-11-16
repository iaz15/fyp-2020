import numpy as np
import os
import rename_files
import data_processing
import matplotlib.pyplot as plt


if __name__ == "__main__":
    input_directory = 'friction_raw_data'
    results_directory = 'results_friction'

    # Get filenames in the input directory
    filenames = rename_files.get_filenames(input_directory)

    # Extract invalid and valid filenames
    invalid_filenames, valid_filenames = rename_files.validate_filenames(filenames)

    # Convert valid filenames into a flat list to be compatible with reading & processing modules
    valid_filenames_flat = [item for sublist in valid_filenames for item in sublist]

    # Read csv files
    dfs_raw_data = data_processing.read_data(valid_filenames_flat, load_location_raw=input_directory)

    ####### Robotic Arm Specific Starts #######
    # Process raw data (position & force) separately
    dfs_position_force_filtered = data_processing.process_force_position(dfs_raw_data)

    # Remove the .csv extension from filenames for matching filenames to be compatible with cof processing module
    matching_filenames = [(item[0].rstrip('.csv'), item[1].rstrip('.csv')) for item in valid_filenames]

    # Combine the force and position data to obtain CoF plots (output key example: data_1)
    dfs_result = data_processing.extract_cof_results(matching_filenames, dfs_position_force_filtered)
    ####### Robotic Arm Specific Ends #######

    for key, df in dfs_result.items():
        df.plot(x='sliding_distance_(mm)', y=['x_force_(N)', 'y_force_(N)', 'z_force_(N)', 'coefficient_of_friction'], title="Graphs Combined: "+str(key))
        df.plot(x='sliding_distance_(mm)', y=['speed_x_(mm_s^-1)', 'speed_y_(mm_s^-1)', 'speed_z_(mm_s^-1)'], title="Graphs Combined: "+str(key))

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='sliding_distance_(mm)', ylabel='coefficient_of_friction', title=str(key))
        ax.scatter(df['sliding_distance_(mm)'],df['coefficient_of_friction'], s=1)
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.show()

    print("")
    print("Checking if results folder exists...")
    data_processing.check_folder_exists(results_directory)
    # create the files
    for idx, (key, value) in enumerate(dfs_result.items(), 1):
        exp_num = key.split("_")[1]
        output_filename = str(exp_num) + "_outputzz.csv"
        output_filepath = os.path.join(results_directory, output_filename)
        dfs_result[key].to_csv(output_filepath, sep=',', index=False)
        print(key)
