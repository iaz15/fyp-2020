import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy import (Table, Column, Integer, Numeric, String, ForeignKey, Float,
                        PrimaryKeyConstraint, UniqueConstraint, CheckConstraint)
from sqlalchemy import insert
from sqlalchemy import func
from sqlalchemy import update
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math


# https://stackoverflow.com/questions/57225904/type-annotating-pandas-dataframes
# https://stackoverflow.com/questions/29221551/can-sphinx-napoleon-document-function-returning-multiple-arguments
def average_datasets(experimental_datasets, exp_numbers=None, trim_resample=True, y_axis='coefficient_of_friction', plot_results=True):
    """
    Given a number of experimental datasets (pandas dataframes)
    this function will interpolate and average the
    coefficient of friction data at different sliding
    distance with its associated values of time. This can be used
    as a standalone function without the database.

    Parameters
    ----------
    experimental_datasets: list
        A list containing a number of pandas datasets (2 or more).
    exp_numbers: list, optional
        A list containing the experiment number associated with each dataframe.
        Both these lists should be in the same order.
    trim_resample: bool, optional
        A flag to indicate whether the averaged results should be trimmed
        and resampled. This is a preset procedure for post processing of the
        data. The default value is True.
    y_axis: str, optional
        The name of the column containing the data that needs to be averaged.
        The default column chosen is the coefficient_of_friction column.
    plot_results: bool, optional
        A flag to indicate whether the results should be plotted. The
        default value is True.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe containing averaged coefficient of friction
        data, sliding distance and standard deviations for each point.
    """

    # Possible values to analyse include:
        # coefficient_of_friction
        # z_force_(N)
        # speed_x_(mm_s^-1)

    # For each of them, make 'time_elapsed_(s)' the index before merging.
    for experimental_dataset in experimental_datasets:
        experimental_dataset.set_index('time_elapsed_(s)', inplace=True)

    # Merge the data, then interpolate in between.
    # Drop null values (Typically at the ends)
    merged_data = pd.concat(experimental_datasets, join='outer', axis=1).interpolate().dropna()


    # Calculate standard deviations for envelops
    standard_deviations = merged_data[y_axis].std(axis=1)

    # Group/split by columns and then average them.
    averaged_data = merged_data.groupby(level=0, axis=1).mean()
    averaged_data['standard_deviations'] = standard_deviations

    # Reset the index of the time column.
    averaged_data.reset_index(inplace=True)

    if trim_resample == True:
        # Maximum to trim off.
        max_sd = 87

        ### Trim the data
        averaged_data = averaged_data[averaged_data['sliding_distance_(mm)'] < max_sd]

        ### Resample the data. These are user presets. TODO: Change these as necessary.
        # Potentially the max_sd could be obtained from the already existing maximum
        # sliding distance if the user doesnt want to specify it.

        # Intervals to use for each data point.
        sd_gap = 1

        sd_list = list(range(0, max_sd, sd_gap))

        sd_list_closest = []
        for sd_interval in sd_list:
            sd_closest = float(averaged_data.iloc[(averaged_data['sliding_distance_(mm)']-sd_interval).abs().argsort()[:1]]['sliding_distance_(mm)'].values)
            sd_list_closest.append(sd_closest)

        averaged_data = averaged_data[averaged_data['sliding_distance_(mm)'].isin(sd_list_closest)]

    # print(averaged_data[y_axis].mean())
    # Now plot the results to check.
    if plot_results == True:
        plt.scatter(averaged_data['sliding_distance_(mm)'], averaged_data[y_axis], label='averaged', s=1)

        # if exp_numbers == None:
        #     for exp_num, dataframe in enumerate(experimental_datasets, 1):
        #         plt.scatter(dataframe['sliding_distance_(mm)'], dataframe[y_axis], label=f'Experiment {exp_num}', s=1)
        # else:
        #     for exp_num, dataframe in zip(exp_numbers, experimental_datasets):
        #         plt.scatter(dataframe['sliding_distance_(mm)'], dataframe[y_axis], label=f'Experiment {exp_num}', s=1)

        print()
        plt.fill_between(averaged_data['sliding_distance_(mm)'],
                         averaged_data[y_axis] - averaged_data['standard_deviations'],
                         averaged_data[y_axis] + averaged_data['standard_deviations'],
                         color='green', alpha=0.2)

        plt.xlabel('sliding_distance_(mm)')
        plt.ylabel(y_axis)
        plt.xlim(0, max(averaged_data['sliding_distance_(mm)']))
        plt.ylim(0)
        plt.grid()
        plt.legend(ncol=2)
        plt.show()

    return averaged_data


def sql_average_datasets(group_id, temperature, speed, force, volume, averaging_col='coefficient_of_friction', processed_load_loc=None, avg_save_loc=None, plot_results=True):
    """
    Given a group, temperature, speed, force and volume this function
    will average the results and produce standard deviations for each
    point. There must be at least 2 valid files for the function to
    work.

    Parameters
    ----------
    group_id: int
        An integer representing the group the user wants to use for
        averaging
    temperature: Union[int, float]
        The temperature the test is carried out at.
    speed: Union[int, float]
        The speed the test is carried out at.
    force: Union[int, float]
        The downwards force being applied during the test.
    volume: Union[int, float]
        The amount of lubricant applied before the test.
    averaging_col: str, optional
        The column of which its values will be used as the y values.
        This column will be averaged.
    processed_load_loc: str, optional.
        The folder or path to the folder containing the processed data
        to be used for averaging. Default value of None indicating
        current location.
    avg_save_loc: str, optional.
        The folder or path where the averaged file should be saved.
        Default value of None indicating current location.
    plot_results: bool, optional
        To plot the result or not. Default value of True.

    Returns
    -------
    None
        Just averages the dataset.
    """

    print("")
    print("Extracting data for averaging...")
    # The processed results folder should be in a folder directly underneath.

    # https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
    np.random.seed(420)

    # Handy tip! TODO: Implement this method in the main code.
    # https://www.youtube.com/watch?v=KM1dtuOkb4Y
    # filenames = ['test.csv', 'potato.csv', 'apples.csv']
    # dataframes = [pd.read_csv(f) for f in filenames]

    # I want to now extract data from the database
    engine = create_engine('sqlite:///friction_practice.db')
    con = engine.connect()
    metadata = db.MetaData()

    # Load the schema for the 1 table required.
    experiments = db.Table('experiments', metadata, autoload=True, autoload_with=engine)

    # Pick where the group is the same and temperature, speed, force and volume are the same
    query = db.select([experiments])\
              .where(experiments.c.group_id == group_id)\
              .where(experiments.c.temperature == temperature)\
              .where(experiments.c.speed == speed)\
              .where(experiments.c.force == force)\
              .where(experiments.c.volume == volume)\
              .where(experiments.c.select == True)

    ResultProxy  = con.execute(query)
    results = ResultProxy.fetchall()

    # Check for the number of results that match.
    num_tests = len(results)
    if num_tests == 0:
        print(f"Number of tests done: 0 for this set of conditions")
        print("Averaging cannot be done, script ending")
        con.close()
        return 0
    elif num_tests == 1:
        print(f"Number of tests done: {num_tests} for this set of conditions")
        print("Experiment ID:")

        df = pd.DataFrame(results)
        columns = results[0].keys()
        df.columns = columns

        print(df['experiment_id'].values)

        print("Averaging cannot be done, script ending")
        con.close()
        return 0
    else:
        # Otherwise there are 2 or more tests, averaging can be done.
        df = pd.DataFrame(results)
        columns = results[0].keys()
        df.columns = columns

        print(f"Number of tests done: {num_tests} for this set of conditions")
        print("Experiment IDs:")
        print(df['experiment_id'].values)


        # Get all the values in file_name and store in a list.
        filenames = df.file_name.values.T.tolist()

        # Now I want to load the files
        if processed_load_loc == None:
            # Assume all files are in the current directory.
            dataframes = [pd.read_csv(f) for f in filenames]
        else:
            # Use the location specified.
            dataframes = [pd.read_csv(f"{processed_load_loc}\\{f}") for f in filenames]

        exp_numbers = list(df['experiment_id'].values)

        # Now process as required.
        df_averaged_data = average_datasets(dataframes, exp_numbers=exp_numbers, plot_results=plot_results, y_axis=averaging_col)

        # Get the file name. Will use the experiment number of the
        # first matching test to determine the filename.
        first_experiment_num = df['experiment_id'][0]
        processed_filename = f"{first_experiment_num}_averaged"

        # Save the results
        if avg_save_loc == None:
            # If not specified location, save in current location.
            df_averaged_data.to_csv(processed_filename, sep=',', index=False)
            print(f"Saved: {processed_filename}")
        else:
            # Otherwise save in specified folder location.

            # Create the folder if it does not exist.
            root = os.getcwd()

            subdir = os.path.join(root, avg_save_loc)

            # Create the folder if it doesn't exist
            if not os.path.exists(subdir):
                print("Creating a new subfolder!")
                os.makedirs(subdir)
            else:
                print("Folder currently exists!")

            df_averaged_data.to_csv(f"{avg_save_loc}\\{processed_filename}.csv", sep=',', index=False)
            print(f"Saved: {avg_save_loc}\\{processed_filename}.csv")

        # Now update the average_file_name & avg_datetime_updated columns
        # Use the experiment_id to determine which rows to update (because unique)

        # https://docs.sqlalchemy.org/en/13/core/dml.html
        today = datetime.datetime.now()
        rows_updated = 0
        for experiment_id in df['experiment_id'].values:
            # Need to convert experiment_id from numpy number to integer to make it work.
            u = update(experiments).where(experiments.c.experiment_id == int(experiment_id))
            u = u.values(avg_file_name=f"{processed_filename}.csv", avg_datetime_updated=today)
            result = con.execute(u)
            rows_updated = rows_updated + result.rowcount

        # TODO: Close the database connection
        con.close()
        print(f"{rows_updated} rows were updated! (In the database)")

    con.close()
    return 1


def extract_data_optimisation(group_id_chosen, avg_load_loc=None):
    """
    Parameters
    ----------
    group_id: int
        An integer representing the group the user wants to use for
        averaging.
    average_loc_loc: str, optional
        The file path to the folder which holds the files containing
        the averaged results.

    Returns
    -------
    None
        Just averages the dataset.
    """
    print("")
    print("Extracting data for optimisation...")
    # Load the database.
    engine = create_engine('sqlite:///friction_practice.db')
    con = engine.connect()
    metadata = db.MetaData()

    # Load the schema for 3 tables that currently exist.
    experiments = db.Table('experiments', metadata, autoload=True, autoload_with=engine)
    optimisation_groups = db.Table('optimisation_groups', metadata, autoload=True, autoload_with=engine)
    lubricants = db.Table('lubricants', metadata, autoload=True, autoload_with=engine)

    # Choose the experiments where there is an averaged file existing & the data entries are not empty.
    columns = [experiments.c.avg_file_name, experiments.c.temperature,
               experiments.c.speed, experiments.c.force,
               experiments.c.volume, experiments.c.film_thickness,
               experiments.c.pressure, optimisation_groups.c.blank_roughness,
               optimisation_groups.c.pin_roughness, lubricants]

    query = db.select(columns)
    query = query.select_from(experiments.join(optimisation_groups.join(lubricants)))\
              .where(experiments.c.group_id == group_id_chosen)\
              .where(experiments.c.avg_file_name != None)\
              .where(experiments.c.film_thickness != None)\
              .where(experiments.c.pressure != None)\
              .where(experiments.c.select == True)\

    ResultProxy  = con.execute(query)
    results = ResultProxy.fetchall()

    if len(results) != 0:
        df = pd.DataFrame(results)
        columns = results[0].keys()
        df.columns = columns

        # Remove duplicates of the same temperature, speed, force and volume
        df = df.drop_duplicates(subset=["temperature", "force", "speed", "volume"])
        print(f"Lubricant: {str(df['name'].values)}")
        if df.any().all() == False:
            # If there are any empty values (False means empty)
            print("Empty Columns:")
            for col in columns:
                # Go through all the columns to determine which is the empty ones.
                if df.any()[col] == False:
                    print(f"    - {col}")

            print(f"Aborting test, fill in the null columns first for lubricant_id {set(df.lubricant_id.values)}.")
            print("")
            return 0

        else:
            # Extract the testing_parameters, put into a dictionary,
            # and add it to this list.
            testing_parameters_set = []

            # Read the csv file and append the required information to
            # these lists.
            cof_measured_set = []
            time_measured_set = []
            sd_measured_set = []

            # Now transform into the required form for optimisation.
            for _, row in df.iterrows():
                # Create a dictionary from test condition
                temp_dictionary = {}
                temp_dictionary['T'] = float(row['temperature'])
                temp_dictionary['F'] = float(row['force'])
                temp_dictionary['P'] = float(row['pressure'])
                temp_dictionary['v'] = float(row['speed'])
                temp_dictionary['V'] = float(row['volume'])
                temp_dictionary['h0'] = float(row['film_thickness'])
                temp_dictionary['eta_0'] = float(row['eta_0'])
                temp_dictionary['Q_eta'] = float(row['Q_eta'])
                temp_dictionary['mu0_lubricated'] = float(row['mu0_lubricated'])
                temp_dictionary['Q_lubricated'] = float(row['Q_lubricated'])
                temp_dictionary['mu0_dry'] = float(row['mu0_dry'])
                temp_dictionary['Q_dry'] = float(row['Q_dry'])
                temp_dictionary['lambda_1'] = float(row['lambda_1'])
                temp_dictionary['lambda_2'] = float(row['lambda_2'])
                temp_dictionary['c'] = float(row['c'])
                temp_dictionary['k_1'] = float(row['k_1'])
                temp_dictionary['k_2'] = float(row['k_2'])
                temp_dictionary['k_3'] = float(row['k_3'])
                temp_dictionary['name'] = str(row['name'])
                temp_dictionary['blank_roughness'] = float(row['blank_roughness'])
                temp_dictionary['pin_roughness'] = float(row['pin_roughness'])

                testing_parameters_set.append(temp_dictionary)

                # We just need the time, sliding distance, coefficient of friction columns
                f = row['avg_file_name']

                # Now load the data as required.
                if avg_load_loc == None:
                    # Assume all files are in the current directory.
                    df_averaged_data = pd.read_csv(f)
                else:
                    # Use the location specified.
                    df_averaged_data = pd.read_csv(f"{avg_load_loc}\\{f}")

                cof_measured_set.append(df_averaged_data['coefficient_of_friction'].values)
                time_measured_set.append(df_averaged_data['time_elapsed_(s)'].values)
                sd_measured_set.append(df_averaged_data['sliding_distance_(mm)'].values)

            # Now convert the cof, time and sd lists to numpy ndarrays
            cof_measured_set = np.array(cof_measured_set)
            time_measured_set = np.array(time_measured_set)
            sd_measured_set = np.array(sd_measured_set)

            return (testing_parameters_set, time_measured_set, sd_measured_set, cof_measured_set)
    else:
        print("No complete results to optimise with")
        return 0

def gui_avg_datasets(group_id_chosen=1, temperature_chosen=20, speed_chosen=80, force_chosen=5, volume_chosen=23.2):
    """ Function to be used with the GUI to average the datasets """

    processed_load_loc = "friction_processed_results"
    avg_save_loc = "friction_averaged_results"

    # Possible values to analyse include:
        # coefficient_of_friction
        # z_force_(N)
        # speed_x_(mm_s^-1)

    return_val = sql_average_datasets(group_id_chosen, temperature_chosen,
                         speed_chosen, force_chosen, volume_chosen,
                         processed_load_loc=processed_load_loc,
                         avg_save_loc=avg_save_loc, plot_results=True,
                         averaging_col='coefficient_of_friction')

    return return_val

if __name__ == "__main__":

    ### Average results for optimisation
    group_id_chosen = 2
    temperature_chosen = 20
    speed_chosen = 80
    force_chosen = 8
    volume_chosen = 23.2

    processed_load_loc = "friction_processed_results"
    avg_save_loc = "friction_averaged_results"

    # Possible values to analyse include:
        # coefficient_of_friction
        # z_force_(N)
        # speed_x_(mm_s^-1)

    return_val = sql_average_datasets(group_id_chosen, temperature_chosen,
                         speed_chosen, force_chosen, volume_chosen,
                         processed_load_loc=processed_load_loc,
                         avg_save_loc=avg_save_loc, plot_results=True,
                         averaging_col='coefficient_of_friction')

    print(return_val)
    # ### Extracting averaged for optimisation
    # # Choose the optimisation group & file loading location
    # group_id_chosen = 1
    # avg_load_loc = "friction_averaged_results"

    # results = extract_data_optimisation(group_id_chosen, avg_load_loc)

    # if results == 0:
    #     print("Optimisation failed, ending script.")
    # else:
    #     # Unpack the variables.
    #     print("Unpacking the variables...")
    #     testing_parameters_set, time_measured_set, sd_measured_set, cof_measured_set = results
    #     num_averaged_results = len(testing_parameters_set)
    #     print(f"There are {num_averaged_results} datasets to optimise")

    #     print(testing_parameters_set)
