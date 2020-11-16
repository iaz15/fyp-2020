import math
import scipy
import time
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter
from scipy.integrate import odeint
from operator import itemgetter
import datetime
import uuid
import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy import (Table, Column, Integer, Numeric, String, ForeignKey, Float,
                        PrimaryKeyConstraint, UniqueConstraint, CheckConstraint, Boolean)
from sqlalchemy import insert
from sqlalchemy import func
from sqlalchemy import update
from prettytable import PrettyTable
import shutil
import re
import sys
# TODO learn venv


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def extract_files_directory(load_location_raw, file_delim=','):
    """
    Gets all the files with a specific extension from the selected
    folder in the current directory. By default it will extract
    csv files.

    Parameters
    ----------
    load_location_raw: str
        The folder name containing the raw data files. It assumes that
        The file is in the same directory and the script (So this is
        just a folder name).
    file_delim: str, optional
        The separator for the file being read. By default it assumes
        that files are csv (',').

    Returns
    -------
    list
        A list containing files of the same file type. By default this
        will contain csv files in a specific folder within the current
        directory.
    """

    try:
        filenames = []
        current_dir = os.getcwd()
        path = os.path.join(current_dir, load_location_raw)
        print(f"Using specified folder path of:\n{path}")

        if file_delim==',':
            # Will use the default of .csv files
            file_delim = '.csv'
        else:
            file_delim = f'{file_delim}'

        for file in os.listdir(path):
            if file.endswith(file_delim):
                filenames.append(file)

        # Then sort them according to number
        filenames = [item for item in sorted_nicely(filenames)]

    except:
        print("Failed to extract files from the raw data folder")
        print(f"Please create a folder called \"{load_location_raw}\"")
        print("Ending program in 2 seconds")
        time.sleep(2)
        sys.exit()
    else:
        # Will return filenames if there is no exception.
        return filenames


def read_data(filenames, load_location_raw, file_delim=','):
    """
    Returns a dictionary of data frames with the filename as the key.
    Default use is with a csv file, but can be changed to tab delimited etc.

    Parameters
    ----------
    filenames: list
        A list containing the names of files that need to be read in.
    load_location_raw: str
        The folder name containing the raw data files. It assumes that
        The file is in the same directory and the script (So this is
        just a folder name).
    file_delim: str, optional
        The separator for the file being read. By default it assumes
        that files are csv (',').


    Returns
    -------
    dict
        A dictionary containing dataframes containing the raw data from
        each file passed in. The key for each dataframe will be the
        name of the file used.
    """
    dfs_raw_data = {}
    current_dir = os.getcwd()
    try:
        for csv_file in filenames:
            path = os.path.join(current_dir, load_location_raw, csv_file)
            df = pd.read_csv(path, header=None, sep=file_delim)

            # Get the csv file name (Remove the .csv)
            base = os.path.basename(csv_file)
            filename = os.path.splitext(base)[0]

            # Extract test conditions from filename
            dfs_raw_data[filename] = df
    except:
        print("Failed to read files from the raw data folder")
        print("Ending program in 2 seconds")
        time.sleep(2)
        sys.exit()

    return dfs_raw_data


def plot_scatter(df, x_axis, y_axis, test_condition=None, s=1, origin_0_0=True):
    """
    Takes in one pandas dataframe and two headers to plot a scatter.
    Titles and labels of the x axis and y axis will be determined by
    chosen dataframe headers by default.

    Call plt.show() after this is called to show the graphs.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing information that needs to be plotted.
    x_axis: str
        A string with the label of the column that needs to be plotted
        on the x axis.
    y_axis: str
        A string with the label of the column that needs to be plotted
        on the y axis.
    test_condition: str, optional
        A string containing the title for the graph being plotted.
    s: int, optional
        Specifies the size of each point being plotted.
    origin_0_0: bool, optional
        By default, it will make xlim and ylim at the left and bottom
        0, 0. Set this to False to disable this function.

    Returns
    -------
    None
    """
    _, ax = plt.subplots()
    ax.scatter(df[x_axis], df[y_axis], s=s)
    if test_condition == None:
        ax.set_title(f"{y_axis} vs {x_axis}")
    else:
        ax.set_title(f"{y_axis} vs {x_axis}\n{test_condition}")

    if origin_0_0==True:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    ax.set_xlabel(f"{x_axis}")
    ax.set_ylabel(f"{y_axis}")

    return None


def plot_scatter_select_points(df, x_axis, y_axis1, y_axis2, y_axis3, test_condition=None, s=1, selected_points=3):
    """
    Extends the plot_scatter function. Adds the function to select points.
    Returns a list containing tuples of coordinates in the form of (x, y).

    No need to call plt.show() as it will automatically show the graphs for you with ginput.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing information that needs to be plotted.
    x_axis: str
        The heading of the column containing data to be plotted on the x axis.
    y_axis1: str
        The heading of the column containing data to be plotted on the first
        y axis.
    y_axis2: str
        The heading of the column containing data to be plotted on the second
        y axis.
    y_axis2: str
        The heading of the column containing data to be plotted on the third
        y axis.
    test_condition: str, optional
        Does nothing. Obsolete.
    s: int
        Specifies the size of the points plotted
    test_condition=None, s=1, selected_points=3
    Returns
    -------
    list
        A list containing files of the same file type. By default this
        will contain csv files in a specific folder within the current
        directory.
    """
    # The default for the number of points to select is 3, but you can change it.
    plot_scatter_multiple(df, x_axis, y_axis1, y_axis2, y_axis3, test_condition=test_condition, s=1)
    points = plt.ginput(selected_points, timeout=0)

    # For some reason the plots dont close until the end, so I forcibly close the plots each time.
    plt.close()

    return points


def plot_scatter_multiple(df, x_axis, y_axis1, y_axis2, y_axis3, test_condition=None, s=1):
    """
    Plots a graph with three y axes and a shared x axis.
    Takes in a dataframe, label of the x axis and three y axes. You have to input all of these.
    Refer to https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales.
    Because the minimum ylim values are set to 0 because for scientific purposes this should be the case,
    if negative values are used this function must be modified.

    You need to call plt.show() after calling this function to show the results.

    Parameters
    ----------
    df: pd.Dataframe
        A pandas dataframe containing the data that needs to be plotted.
    x_axis: str
        The title of the column with data to be used on the x axis of the plot.
    y_axis1: str
        The title of the column with data to be used for the first y axis.
    y_axis2: str
        The title of the column with data to be used for the second y axis.
    y_axis3: str
        The title of the column with data to be used for the third y axis.
    test_condition: str, optional
        Does nothing.
    s: int, optional
        Changes the size of points plotted.

    Returns
    -------
    None
        Just plots the data.

    """
    fig = plt.figure()
    host = fig.add_subplot(111)

    # Create the extra y axes
    par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlabel(x_axis)
    host.set_ylabel(y_axis1)
    par1.set_ylabel(y_axis2)
    par2.set_ylabel(y_axis3)

    host_max_ylim = 1.2 * max(df[y_axis1])
    par1_max_ylim = 1.2 * max(df[y_axis2])
    par2_max_ylim = 1.2 * max(df[y_axis3])

    host_max_xlim = 1.01 * max(df[x_axis])

    host.set_xlim(0, host_max_xlim)
    host.set_ylim(0, host_max_ylim)
    par1.set_ylim(0, par1_max_ylim)
    par2.set_ylim(0, par2_max_ylim)

    # 12 indicates the list of colours used to define the colourmap
    # https://matplotlib.org/tutorials/colors/colormap-manipulation.html
    viridis = cm.get_cmap('viridis', 12)
    color1 = viridis(0.3)
    color2 = viridis(0.5)
    color3 = viridis(0.9)

    # Having commas in front of p1, etc is necessary - Good idea to find out why
    p1, = host.plot(df[x_axis], df[y_axis1], color=color1,label=y_axis1)
    p2, = par1.plot(df[x_axis], df[y_axis2], color=color2, label=y_axis2)
    p3, = par2.plot(df[x_axis], df[y_axis3], color=color3, label=y_axis3)
    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')

    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 60))

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    plt.tight_layout()


def plot_scatter_multiple_flex(df, x_axis, y_axes_list, test_condition=None, s=1):
    """
    Based on the plot_scatter function. There needs to be at least 2 plots in the y axes_list to work
    Takes in an array of y axes, e.g. ['coefficient_of_friction', 'x_force_(N)'].

    You need to call plt.show() after calling this function to show the results.

    Parameters
    ----------
    df: pd.Dataframe
        A dataframe containing data that the user wants to plot.
    x_axis: str
        The title of the column of the data to be used on the x axis.
    y_axes_list: list
        A list of strings containing the titles of the columns to be
        plotted on the y axis. There can be any amount
    test_condition: str, optional
        Does nothing.
    s: int, optional
        Changes the size of points plotted.

    Returns
    -------
    None
        Just plots the data.
    """
    # Initialise the host
    fig = plt.figure()
    host = fig.add_subplot(111)
    host.set_xlabel(x_axis)
    host.set_ylabel(y_axes_list[0])

    number_y_axes = len(y_axes_list)

    twin_axes_dict = {}
    # Initialise the twin axes/pars (par1, par2, etc)
    twinx_name = "par"
    for extra_axis in range(1, number_y_axes):
        # When you want 3/4 of the total y axes
        twin_axes_dict[twinx_name+str(extra_axis)] = host.twinx()
        twin_axes_dict[twinx_name+str(extra_axis)].set_ylabel(y_axes_list[extra_axis])

    # Initialise the colours (colour1, colour2, etc) and the plots (p1, p2, etc)
    colours_dict = {}
    plot_dict = {}

    # https://matplotlib.org/tutorials/colors/colormap-manipulation.html
    viridis = cm.get_cmap('viridis', 12)
    colour_division = 1/number_y_axes

    lns = []
    for y_axis in range(1, (number_y_axes+1)):
        # Starts from 1 until the last number of y axes
        # When you want 4/4 of the total y axes
        colours_name = "colour"
        plot_names = "p"

        colours_dict[colours_name+str(y_axis)] = viridis(colour_division*(y_axis))

        if y_axis==1:
            # Initialise for host
            p1, = host.plot(df[x_axis], df[y_axes_list[0]], color=colours_dict[colours_name+str(y_axis)], label=y_axes_list[0])
            host.yaxis.label.set_color(p1.get_color())
            lns.append(p1)
        else:
            plot_dict[plot_names+str(y_axis)], = twin_axes_dict[twinx_name+str(y_axis-1)].plot(df[x_axis], df[y_axes_list[y_axis-1]], color=colours_dict[colours_name+str(y_axis)], label=y_axes_list[y_axis-1])
            twin_axes_dict[twinx_name+str(y_axis-1)].yaxis.label.set_color(plot_dict[plot_names+str(y_axis)].get_color())
            lns.append(plot_dict[plot_names+str(y_axis)])

    host.legend(handles=lns, loc='best')

    if number_y_axes>2:
        multiple = 60
        for num in range(2, number_y_axes):
            twinx_name = "par"
            twin_axes_dict[twinx_name+str(num)].spines['right'].set_position(('outward', multiple*(num-1)))

    plt.tight_layout()


def plot_scatter_xyz(df, x_axis, y_axis, s=1, title=None):
    """
    Plots the x, y, z components of any data on 1 graph.
    The input force/speed/position label must be the x version (e.g. x_force_(N)).

    Call plt.show() after this is called to show the graphs.

    Parameters
    ----------
    df: pd.Dataframe
        A dataframe containing data that the user wants to plot.
    x_axis: str
        The title of the column of the data to be used on the x axis.
    y_axis: str
        The title of the column of the data to be used on the y axis.
    s: int, optional
        Specifies the size of the points being plotted.
    title: str, optional
        Specifies the title of the plot.

    Returns
    -------
    None
        Just averages the dataset.
    """
    _, ax = plt.subplots()

    directions = ["x", "y", "z"]
    y_axes = []
    for direction in directions:
        y_axes.append(y_axis.replace("x ", direction + " "))

    for y_axis in y_axes:
        ax.scatter(df[x_axis], df[y_axis], s=s, label=y_axis)

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axes[0].replace("x ", ""))


def plot_force_scatter(df):
    """
    Plots a scatter graph of the x, y, z force data of the robotic arm over time given the correct dataframe.

    Call plt.show() after this is called to show the graphs.

    Parameters
    ----------
    df: pd.Dataframe
        A dataframe containing data that the user wants to plot.

    Returns
    -------
    None
        Just plots the data.
    """

    plot_scatter_xyz(df, x_axis='time_(s)', y_axis='x_force_(N)', title="Force sensor measurements")


def plot_speed_scatter(df):
    """
    Plots a scatter graph of the x, y, z speed of from the robotic arm over time given the correct dataframe.
    Note that speed may not be accurate.

    Call plt.show() after this is called to show the graphs.

    Parameters
    ----------
    df: pd.Dataframe
        A dataframe containing data that the user wants to plot.

    Returns
    -------
    None
        Just plots the data.
    """

    plot_scatter_xyz(df, x_axis='time_(s)', y_axis='speed_x_(mm_s^-1)', title="Speed of the robot arm in x, y and z directions")


def filter_position_data(df):
    """
    Filtering conditions for position data.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing position data.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the only the relevant test data.
        Other data has been filtered according to set rules.
    """
    filtered_speed_range = df[(df['speed_x_(mm_s^-1)'] < 1000) &
                              (df['speed_x_(mm_s^-1)'] > 20) &
                              (df['speed_y_(mm_s^-1)'] < 5) &
                              (df['speed_z_(mm_s^-1)'] < 5)]
    return filtered_speed_range


def process_position_data(df, filter_data=True):
    """
    Processes the raw position data.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the raw position data.
    filter_data: bool, optional
        A flag to indicate if the data should be filtered (Data collected
        outside testing will automatically be removed otherwise).

    Returns
    -------
    pd.DataFrame
        A dataframe that only contains the relevant data.
    """

    # Assign the column names of the data.
    df.columns = ['time_(s)', 'x_position_(m)', 'y_position_(m)', 'z_position_(m)']

    # Get delta t
    df['delta_t_(s)'] = df['time_(s)'].diff().fillna(0)

    # For position data, it is moving along the -ve x direction during the test
    df['delta_x_(m)'] = df['x_position_(m)'].diff().fillna(0)
    df['speed_x_(mm_s^-1)'] = df['delta_x_(m)'].divide(df['delta_t_(s)']).multiply(1000).abs()

    df['delta_y_(m)'] = df['y_position_(m)'].diff().fillna(0)
    df['speed_y_(mm_s^-1)'] = df['delta_y_(m)'].divide(df['delta_t_(s)']).multiply(1000).abs()

    df['delta_z_(m)'] = df['z_position_(m)'].diff().fillna(0)
    df['speed_z_(mm_s^-1)'] = df['delta_z_(m)'].divide(df['delta_t_(s)']).multiply(1000).abs()

    # Convert positions in m to mm
    df['x_position_(m)'] = df['x_position_(m)'].multiply(1000)
    df['y_position_(m)'] = df['y_position_(m)'].multiply(1000)
    df['z_position_(m)'] = df['z_position_(m)'].multiply(1000)

    # Its not done directly because for some reason it gives an error.
    df.rename(columns={'x_position_(m)': 'x_position_(mm)',
                        'y_position_(m)': 'y_position (mm)',
                        'z_position_(m)': 'z_position_(mm)'}, inplace=True)

    # Now extract the relevant data (When the test has started)
    # If the speed is been 30 mm/s +- 5 & y and x speeds are low, it is in the correct region.

    ### Extract the required data ###
    if filter_data == True:
        filtered_speed_range = filter_position_data(df)
        # As data fluctuates a lot, extract the data from the main file where from the first time to the last time.
        # Filtering using the current method may not be reliable since sometimes the speed drops outside of the range.
        start_index, end_index = filtered_speed_range.index[0], filtered_speed_range.index[-1]
        required_data = df.iloc[start_index:(end_index + 1)]
    else:
        required_data = df.copy()

    # As data fluctuates a lot, extract the data from the main file where from the first time to the last time.
    required_data = required_data[['time_(s)','speed_x_(mm_s^-1)', 'x_position_(mm)']]

    return required_data


def filter_force_data(df):
    """
    Filtering conditions for force data.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing force data.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the only the relevant test data.
        Other data has been filtered according to set rules.
    """
    processed_data = df[((df['z_force_(N)']) > 3) &
                         (df['coefficient_of_friction'] > 0.1)].copy()
    return processed_data


def process_force_data(df, filter_data=True):
    """
    Takes in a dataframe containg raw force data, extracts the required results.
    Returns a dataframe containing: timestamp, delta t,  coefficient_of_friction,
    x, y and z forces.

    If filter_data=True then it will filter based on user set conditions.
    Otherwise it will process the whole dataframe

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing raw force data.
    filter_data: bool, optional
        A flag to indicate if the data should be filtered (Data collected
        outside testing will automatically be removed otherwise).

    Returns
    -------
    pd.DataFrame
        A dataframe containing the processed force data.
    """

    df.columns = ['time_(s)', 'x_force_(N)', 'y_force_(N)', 'z_force_(N)']
    df['delta_t_(s)'] = df['time_(s)'].diff().fillna(0)

    # Calculate coefficient_of_friction
    df['coefficient_of_friction'] = (df['y_force_(N)'])/(df['z_force_(N)'])

    # Convert z force to +ve values
    df['z_force_(N)'] = abs(df['z_force_(N)'])

    ### Extract the required data ###
    if filter_data == True:
        filtered_force_range = filter_force_data(df)
        start_index, end_index = filtered_force_range.index[0], filtered_force_range.index[-1]
        processed_data = df.iloc[start_index:(end_index + 1)]
    else:
        processed_data = df.copy()

    start_time = processed_data['time_(s)'].iloc[0]
    end_time = processed_data['time_(s)'].iloc[-1]
    test_time = end_time - start_time

    max_cof = processed_data['coefficient_of_friction'].max()

    # print(f"\t- Testing time: {round(test_time, 3)} s")
    # print(f"\t- Maximum coefficient_of_friction: {round(max_cof, 2)}")

    return processed_data


def find_matching_csv(processed_data_keys):
    """
    Given a list of all the filenames obtained from keys in the dictionary
    storing all the extracted data by filenames, matches the corresponding force
    and position filenames and returns a list containing tuples of the
    matching filenames.

    Parameters
    ----------
    processed_data_keys: list
        A list containing filenames of all files containing the
        force and position data processed.

    Returns
    -------
    list
        A list containing tuples of matching files for force and position
        data. Each tuple with be 2 in length.
    """

    # First we need to split the data into its two parts
    # if it matches another one, with just force swapped with position or vice versa match it.
    matching_filenames = []

    # The first word is the identifier. Anything before that is extra
    for item in processed_data_keys:
        try:
            temp_split = item.split("_")
            if "force" in temp_split[1]:
                joined_temp_split = "_".join(temp_split)
                swap_force_position = joined_temp_split.replace("force", "position")
                if swap_force_position in processed_data_keys:
                    processed_data_keys.remove(swap_force_position)
                    matching_filenames.append((item, swap_force_position))
                else:
                    print(f"File is missing its matching position file:\n\t{item}")

            elif "position" in temp_split[1]:
                joined_temp_split = "_".join(temp_split)
                swap_position_force = joined_temp_split.replace("position", "force")
                if swap_position_force in processed_data_keys:
                    processed_data_keys.remove(swap_position_force)
                    matching_filenames.append((item, swap_position_force))
                else:
                    print(f"File is missing its matching force file:\n\t{item}")

            else:
                print(f"An invalid file was passed through:\n\t{item}")
        except:
            print(f"An invalid file was passed through:\n\t{item}")

    return matching_filenames


def extract_cof_results(matching_files, dfs_dictionary, pick_points=True):
    """
    This is the last step.
    It will return only the data the user wants from selecting the points.

    Parameters
    ----------
    matching_files: list
        A list of all the matching files stored in tuples. Each tuple
        contains two strings (matching force and position file names).
    dfs_dictionary: dict
        A dictionary containing
    pick_points: bool, optional
        A flag to indicate whether the user wants to pick points to process
        the data in sections or not.

    Returns
    -------
    dict
        A dictionary containing all the required information (in a dataframe)
        for fitting later. The key of each dataframe will be the file name.
    """
    dfs_output = {}

    for idx, match in enumerate(matching_files, 1):
        if "force" in match[0]:
            force_label = match[0]
            position_label = match[1]
        elif "position" in match[0]:
            # Normally force always comes first before position because files are stored in alphabetical order.
            # This is done just in case.
            force_label = match[0]
            position_label = match[1]
        else:
            raise Exception('force or position should be in the filename used')

        # Get the test conditions to label data. This is just some string manipulation.
        key_label = match[0].replace("force_", "").replace("position_", "")

        # print(f"Processing File {idx}")
        # print(f"File: {key_label}")
        df_force = dfs_dictionary[force_label].copy()
        df_position = dfs_dictionary[position_label].copy()

        df_force['time_elapsed_(s)'] = df_force['time_(s)'].subtract(df_force['time_(s)'].iloc[0])
        df_position['time_elapsed_(s)'] = df_position['time_(s)'].subtract(df_position['time_(s)'].iloc[0])

        # Making the time since test started the index column (for interpolation)
        df_force.set_index('time_elapsed_(s)', inplace=True)
        df_position.set_index('time_elapsed_(s)', inplace=True)

        # Merge the two dataframes (for interpolation)
        df = pd.merge(df_force, df_position, right_index=True, left_index=True, how='outer')

        # Interpolate NaN values for columns we want (The ones we dont interpolate have NaN values)
        df['coefficient_of_friction'].interpolate(inplace=True)
        df['x_position_(mm)'].interpolate(inplace=True)
        df['x_force_(N)'].interpolate(inplace=True)
        df['speed_x_(mm_s^-1)'].interpolate(inplace=True)
        df['z_force_(N)'].interpolate(inplace=True)
        df['y_force_(N)'].interpolate(inplace=True)

        # Now we can reset the index since interpolation is done
        df.reset_index(inplace=True)

        # Drop duplicates caused by interpolation
        df.drop_duplicates(subset='speed_x_(mm_s^-1)', inplace=True)

        # Calculate sliding distance
        df['delta_t_(s)'] = df['time_elapsed_(s)'].diff().fillna(0)
        df['delta_x_(mm)'] = df['speed_x_(mm_s^-1)'].multiply(df['delta_t_(s)'], axis='index')
        df['sliding_distance_(mm)'] = df['delta_x_(mm)'].cumsum()

        # Always remove these columns
        df.drop(['time_(s)_x', 'time_(s)_y', 'delta_t_(s)',\
                 'delta_x_(mm)'], axis=1, inplace=True)

        if pick_points==True:
            # Plotting the results, and extracting the points which the user wants.
            # Assume that the first point is at sliding distance = 0 (Nothing wrong with initial point)
            selected_points = plot_scatter_select_points(df, "sliding_distance_(mm)", "speed_x_(mm_s^-1)", "z_force_(N)", "coefficient_of_friction",
                                                        test_condition=key_label, selected_points=4)

            # It is set to (0, 0) because only the x point is relevant (Needs to be set to 0), the y can be anything.
            # Creating an initial starting point.
            selected_points.insert(0, (0, 0))

            # Arrange the points by ascending x values if points are not clicked in the right order.
            selected_points = sorted(selected_points, key=itemgetter(0))

            segment_tuples = []
            for i in range(len(selected_points)-1):
                # These are the x values (In this case sliding distances) of the different segments.
                segment_tuples.append((selected_points[i][0], selected_points[i+1][0]))

            print("")
            print(f"Number of segments: {len(segment_tuples)}")
            for count, segment in enumerate(segment_tuples):

                df_segment = df[(df['sliding_distance_(mm)'] >= segment[0]) & (df['sliding_distance_(mm)'] < segment[1])].copy()
                # Then do filtering as desired on each segment
                # Using assign seems to stop the warning asking the user to use .copy() from popping up in the console
                # Assign: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html
                # Savgol Filter: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
                print(f"There are {len(df_segment.index)} points in the segment you chose")
                while True:
                    try:
                        window_length = int(input(f'Enter a window length {count+1}/{len(segment_tuples)}. It must be odd and < {len(df_segment.index)}\n'))
                        if window_length > 1 and window_length%2 != 0 and window_length < len(df_segment.index):
                            break
                        else:
                            print("Please enter a number > 1 and < ")
                            continue
                    except:
                        print("Please enter a number > 1")
                        continue

                # First do the smoothing
                df_segment['coefficient_of_friction'] = savgol_filter(df_segment['coefficient_of_friction'].values, window_length, 2)

                print(f"There are {len(df_segment.index)} points in the segment you chose")
                # Ask the user how much resampling to be done on each part
                while True:
                    try:
                        resample_amount = int(input(f'Enter a resampling ratio {count+1}/{len(segment_tuples)}\n'))
                        if resample_amount > 0:
                            break
                        else:
                            print("Please enter a number > 0")
                            continue
                    except:
                        print("Please enter a number > 0")
                        continue

                print()
                # Then do the resampling
                df_segment = df_segment.iloc[::resample_amount, :]

                # Then append to dataframe of all data
                if key_label not in dfs_output:
                    dfs_output[key_label] = df_segment
                else:
                    dfs_output[key_label] = dfs_output[key_label].append(df_segment, ignore_index=True)
        else:
            window_length = 51
            min_time = 0.06

            df = df[df['time_elapsed_(s)'] > min_time]
            print(len(df['coefficient_of_friction'].values))
            print(force_label)
            df['coefficient_of_friction'] = savgol_filter(df['coefficient_of_friction'].values, 51, 2)
            dfs_output[key_label] = df

    return dfs_output


def extract_conditions(filename):
    """
    Given a filename of the of a format:

    1_force_lubricantid=1_pinMat=P20_pinRa=0.8_blankMat=AA7075_blankRa=0.3_T=350_F=5_v=50_V=23.2

    will return dictionaries containing the required test condition
    information.

    Parameters
    ----------
    filename: str
        An string which is the name of the file containing the raw data.
        It will contain details of the test conditions.

    Returns
    -------
    test_conditions_group: dict
        A dictionary containing test conditions for grouping.
        (lubricant_id, pin_material, pin_roughness, blank_material, blank_roughness).
    experiment_summary: dict
        A dictionary with the test conditions for model fitting.
        (T, F, v and V).
    """
    temp = filename.split("_")
    test_conditions_group = {}
    experiment_summary = {}

    for i in temp:
        if "T" in i:
            experiment_summary["temperature"] = float(i.split("=")[1])
        if "F" in i:
            experiment_summary["force"] = float(i.split("=")[1])
        if "v" in i:
            experiment_summary["speed"] = float(i.split("=")[1])
        if "V" in i:
            experiment_summary["volume"] = float(i.split("=")[1])
        if "lubricant" in i:
            test_conditions_group["lubricant_id"] = int(i.split("=")[1])
        if "pinMat" in i:
            test_conditions_group["pin_material"] = i.split("=")[1]
        if "pinRa" in i:
            test_conditions_group["pin_roughness"] = float(i.split("=")[1])
        if "blankMat" in i:
            test_conditions_group["blank_material"] = i.split("=")[1]
        if "blankRa" in i:
            test_conditions_group["blank_roughness"] = float(i.split("=")[1])

    return test_conditions_group, experiment_summary


def database_setup(remove_existing=False):
    """
    Sets up the database.

    Parameters
    ----------
    remove_existing: bool
        A boolean indicating where the currently existing database should be deleted.
        This should typically be ALWAYS set to False except for use during testing
        otherwise all the data will be lost.

    Returns
    -------
    None
    """
    # Remove the database file each time.

    if remove_existing==True:
        # Will return an error if the file doesn't exist or it
        # is currently open.
        try:
            os.remove("friction_practice.db")
            print("File Removed!\n")
        except:
            print("An error was encountered when removing the file!\n")

    engine = create_engine('sqlite:///friction_practice.db')
    con = engine.connect()
    metadata = db.MetaData()

    # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91
    # https://docs.sqlalchemy.org/en/13/core/type_basics.html
    # MetaData is used to tie together the database.
    # It is a catalog of Table objects. These tables can be accessed through a dictionary.
    # The dictionary: MetaData.tables.

    # Column is set as the primary key. This is setting up a sql table
    # PRIMARY TABLE: experiments
    experiments = Table('experiments', metadata,
    Column('experiment_id', Integer(), primary_key=True, autoincrement=True),
    Column('file_name', String(200), index=True),
    Column('replicate', Integer()),
    Column('exp_datetime', String(100)),
    Column('group_id', Integer(), ForeignKey('optimisation_groups.group_id')),
    Column('temperature', Integer()),
    Column('speed', Integer()),
    Column('force', Integer()),
    Column('volume', Float()),
    Column('film_thickness', Float()),
    Column('pressure', Float()),
    Column('avg_file_name', String(200), index=True),
    Column('avg_datetime_updated', String(200)),
    Column('select', Boolean(), default=True),
    Column('comments', String(200))
    )

    optimisation_groups = Table('optimisation_groups', metadata,
    Column('group_id', Integer(), primary_key=True, autoincrement=True),
    Column('lubricant_id', Integer(), ForeignKey('lubricants.lubricant_id')),
    Column('blank_material', String(55)),
    Column('blank_roughness', Float()),
    Column('pin_material', String(55)),
    Column('pin_roughness', Float()),
    )

    lubricants = Table('lubricants', metadata,
    Column('lubricant_id', Integer(), primary_key=True, autoincrement=True),
    Column('name', String(55), index=True, nullable=False),

    Column('eta_0', Float(), nullable=False),
    Column('Q_eta', Float(), nullable=False),
    Column('mu0_lubricated', Float(), nullable=False),
    Column('Q_lubricated', Float(), nullable=False),
    Column('mu0_dry', Float(), nullable=False),
    Column('Q_dry', Float(), nullable=False),

    Column('lambda_1', Float(), nullable=False),
    Column('lambda_2', Float(), nullable=False),
    Column('c', Float(), nullable=False),
    Column('k_1', Float(), nullable=False),
    Column('k_2', Float(), nullable=False),
    Column('k_3', Float(), nullable=False)
    )

    # https://towardsdatascience.com/sqlalchemy-python-tutorial-79a577141a91

    metadata.create_all(engine)

    # Add sample existing lubricants. Required, since the user will
    # be getting the lubricant_id from this table each time.
    try:
        # Insert some preset lubricants into the database

        sample_lubricants = dict(lubricant_id = [1, 2],
                                 eta_0 = [0.12, 0.12],
                                 Q_eta = [11930, 11930],
                                 mu0_lubricated = [1.6907, 1.6907],
                                 Q_lubricated = [9141.5068, 9141.5068],
                                 mu0_dry = [10.9422, 10.9422],
                                 Q_dry = [9368.8512, 9368.8512],
                                 lambda_1 = [20, 20],
                                 lambda_2 = [1.1, 1.1],
                                 c = [750, 0.012],
                                 k_1 = [2.05, 2.05],
                                 k_2 = [2.98, 2.98],
                                 k_3 = [4.2, 5.3],
                                 name = ["Lubricant1", "Lubricant2"]
                                )

        sample_lubricants = pd.DataFrame(sample_lubricants)
        sample_lubricants.to_sql('lubricants', engine, if_exists='append', index=False)
    except:
        # It probably exists, don't do anything. It will give an
        # error if it exists because of the unique constraint on lubricant_id
        pass

    # Close the connection to the database.
    con.close()
    return None


def store_results_sql(dfs_result):
    """
    Stores the results in the SQL table.

    Parameters
    ----------
    dfs_result: dict
        A dictionary containing all the required information (in a dataframe)
        for fitting later. The key of each dataframe will be the file name.

    Returns
    -------
    None
    """

    # The user will need to pick the lubricant from the database.
    # The lubricant must be created in the lubricant table before carrying out the test.
    engine = create_engine('sqlite:///friction_practice.db')
    con = engine.connect()
    metadata = db.MetaData()

    # Load the 3 tables that currently exist.
    experiments = db.Table('experiments', metadata, autoload=True, autoload_with=engine)
    optimisation_groups = db.Table('optimisation_groups', metadata, autoload=True, autoload_with=engine)
    lubricants = db.Table('lubricants', metadata, autoload=True, autoload_with=engine)

    # Write the csv file containing experimental data

    processed_results_folder = "friction_processed_results"
    root = os.getcwd()

    subdir = os.path.join(root, processed_results_folder)

    # Create the folder if it doesn't exist
    if not os.path.exists(subdir):
        print("Creating a new subfolder!")
        os.makedirs(subdir)

    for idx, key in enumerate(dfs_result.keys()):

        # Extract the required information from the file name.
        test_conditions_group, experiment_summary = extract_conditions(key)

        ### TEMPORARY ### TODO: REMOVE LATER ### Otherwise the user will manually have to input before proceeding

        # Set the film thickness and pressures to a value.
        experiment_summary['film_thickness'] = 25
        experiment_summary['pressure'] = 0.34

        ################# TODO: REMOVE LATER ### Otherwise the user will manually have to input before proceeding

        print(f"Storing file {idx+1}")

        ### Query for optimisation_groups table #2
        lubricant_id = test_conditions_group['lubricant_id']
        blank_material = test_conditions_group['blank_material']
        blank_roughness = test_conditions_group['blank_roughness']
        pin_material = test_conditions_group['pin_material']
        pin_roughness = test_conditions_group['pin_roughness']

        # First query if there is this combination in optimisation_groups
        query = db.select([optimisation_groups.c.group_id])\
                         .where(optimisation_groups.c.lubricant_id == lubricant_id)\
                         .where(optimisation_groups.c.blank_material == blank_material)\
                         .where(optimisation_groups.c.blank_roughness == blank_roughness)\
                         .where(optimisation_groups.c.pin_material == pin_material)\
                         .where(optimisation_groups.c.pin_roughness == pin_roughness)

        ResultProxy  = con.execute(query)
        matching_group_num = ResultProxy.scalar()

        if matching_group_num == None:
            # Find max group_id
            query = db.select([func.max(optimisation_groups.c.group_id)])
            ResultProxy  = con.execute(query)
            max_group_id = ResultProxy.scalar()

            if max_group_id == None:
                # This means that there are no groups in the table
                new_group_id = 1
            else:
                # As group_id will autoincrement the new value will be + 1
                new_group_id = max_group_id + 1

            # Update the SQL table
            experiment_group_df = pd.DataFrame([test_conditions_group])
            experiment_group_df.to_sql('optimisation_groups', engine, if_exists='append', index=False)
            print("Updated optimisation_groups SQL table!")

            # Add this new group_id to the experiments_summary dictionary
            experiment_summary['group_id'] = new_group_id

        else:
            # There is an existing matching group_id
            # Add this matching group_id to the experiments_summary dictionary
            experiment_summary['group_id'] = matching_group_num

        ### Query for experiments table #1 (Main table)

        # T, speed, F, V, group_id must be used to check
        # Generate the current datetime
        today = datetime.datetime.now()

        # Update the dictionary with current datetime
        experiment_summary['exp_datetime'] = today

        # Set the select column to the default of 1 (Use)
        experiment_summary['select'] = 1

        temperature = experiment_summary['temperature']
        speed = experiment_summary['speed']
        volume = experiment_summary['volume']
        force = experiment_summary['force']
        group_id = experiment_summary['group_id']

        # Check for replicate experiments
        query = db.select([func.count()]).where(experiments.c.temperature == temperature)\
                                         .where(experiments.c.speed == speed)\
                                         .where(experiments.c.volume == volume)\
                                         .where(experiments.c.force == force)\
                                         .where(experiments.c.group_id == group_id)

        ResultProxy  = con.execute(query)
        replicate_num = ResultProxy.scalar()

        # Update the dictionary with replicate number
        experiment_summary['replicate'] = replicate_num

        # Get max experiment number to know how to label the file
        # The current experiment = max number + 1
        query = db.select([func.max(experiments.c.experiment_id)])
        ResultProxy  = con.execute(query)
        max_experiment_id = ResultProxy.scalar()

        if max_experiment_id is None:
            current_exp_id = 1
        else:
            current_exp_id = max_experiment_id + 1

        # Filename should be like this:
        # 1_L1_AA7075_P20_rep0
        # Experiment number -> Lubricant number -> Blank material
        # -> Pin material -> Replicate number.
        output_filename = str(current_exp_id) + "_" + "L" + str(lubricant_id) + \
                          "_" + blank_material + "_" + pin_material + \
                          "_" + "rep" + str(replicate_num) + ".csv"

        experiment_summary["file_name"] = output_filename

        output_filepath = os.path.join(subdir, output_filename)
        dfs_result[key].to_csv(output_filepath, sep=',', index=False)

        print(f"Saved the data with the filename: {output_filename}")

        # Update the experiments SQL table
        experiment_summary_df = pd.DataFrame([experiment_summary])
        experiment_summary_df.to_sql('experiments', engine, if_exists='append', index=False)
        print("Updated experiments SQL table!")

        print()


def check_folder_exists(folder_path):
    """
    Checks if the folder exists. If the folder doesn't exist, it is
    created.

    Parameters
    ----------
    folder_path: str
        An string with the name of the folder that is being checked
        to see if it exists.

    Returns
    -------
    None
    """
    if os.path.exists(folder_path):
        print("- Folder exists!")
    else:
        print("- Folder doesn't exist! Creating a new folder.")
        try:
            os.mkdir(folder_path)
        except OSError:
            print (f"Creation of the directory {folder_path} failed")
        else:
            print (f"Created folder: {folder_path} ")

    return None


def process_force_position(dfs_raw_data):
    """
    Parameters
    ----------
    dfs_raw_data: dict
        A dictionary containing pandas DataFrames with keys as the file
        names. Testing raw data (position + force) is stored in each DataFrame.

    Returns
    -------
    dict
        A dictionary of the same format as the input dictionary except now
        the raw data (force + position) in each dataframe has been
        processed and filtered as required.
    """
    # Initialise the dictionary containing the full processed data (Before user input)
    dfs_position_force_filtered = {}
    count = 0

    for file_name, df in dfs_raw_data.items():

        try:
            if "force" in file_name:
                count = count + 1
                print(f"Files processed: {count}", end='\r')


                # Filtered Data
                force_data = process_force_data(df, filter_data=True)
                dfs_position_force_filtered[file_name] = force_data

            elif "position" in file_name:
                count = count + 1
                print(f"Files processed: {count}", end='\r')

                # Filtered Data
                position_data = process_position_data(df, filter_data=True)
                dfs_position_force_filtered[file_name] = position_data

            else:

                print(f"Incorrect File Passed Through.\n")

        except:

            print(f"Incorrect File Passed Through.\n")

    print(f"Files processed: {count}")

    return dfs_position_force_filtered


def main():
    """ Main Function for running the script. """
    t0 = time.time()

    ### USER INPUT ###
    # Indicate the relative location of raw data and bins folders
    raw_data_folder = "friction_raw_data"
    bin_folder = "friction_raw_data_bin"
    ### USER INPUT ###

    startup_text = "Processing Files for Interactive Friction Model."
    print(startup_text)

    ################## Step 1 - Check if the required folders exist ##################

    print("="*len(startup_text))
    print("Checking if the folder for raw data exists...")
    check_folder_exists(raw_data_folder)

    print("Checking if the bin folder exists...")
    check_folder_exists(bin_folder)
    print("="*len(startup_text))

    ################## Step 2 - Extract the csv files to be processed ##################

    csv_files = extract_files_directory(load_location_raw=raw_data_folder)
    dfs_raw_data = read_data(csv_files, load_location_raw=raw_data_folder)
    print("="*len(startup_text))

    ################## Step 3 - Process all the force and position data ##################

    # print("Processing Force and Position Files...")

    dfs_position_force_filtered = process_force_position(dfs_raw_data)

    t1 = time.time()
    time_processing_data = round((t1-t0),4)
    print(f"Time to process force and position data is: {time_processing_data}s")

    ################## Step 4 - Match all the position and force files ##################

    list_keys = list(dfs_position_force_filtered.keys())
    matching_files = find_matching_csv(list_keys)

    ################## Step 5 - Obtain the coefficient_of_friction graphs ################

    # print("Calculating Coefficient of Friction vs Sliding Distance...")

    dfs_result = extract_cof_results(matching_files, dfs_position_force_filtered, pick_points=False)

    ################## Step 6 - Plot the extracted data #################################

    t2 = time.time()
    print(f"Time to calculate coefficient of friction: {round((t2-t1), 4)}s\n")

    print(f"Output file plot:")
    for idx, key_string in enumerate(dfs_result.keys(), 1):
        exp_id = key_string.split("_")[0]
        print(f"Experimental Result {exp_id}")
        test_conditions_group, experiment_summary = extract_conditions(key_string)

        pretty_table_1 = PrettyTable()
        pretty_table_1.field_names = ["Group Test Conditions", "Value/Type"]

        for key, value in test_conditions_group.items():
            pretty_table_1.add_row([key, value])

        pretty_table_2 = PrettyTable()
        pretty_table_2.field_names = ["Experiment Variable", "Value"]

        for key, value in experiment_summary.items():
            pretty_table_2.add_row([key, value])

        print(pretty_table_1) 
        print(pretty_table_2)
        print("")

        ### PLOTTING
        plot_scatter(dfs_result[key_string], 'sliding_distance_(mm)', 'coefficient_of_friction', test_condition=f'Experiment: {exp_id}')
        plt.show()

    ################## Step 7 - Moving the old files into the bin folder #################

    for file1, file2 in matching_files:

        raw_data_path1 = f"{raw_data_folder}\\{file1}.csv"
        raw_data_path2 = f"{raw_data_folder}\\{file2}.csv"

        bin_path1 = f"{bin_folder}\\{file1}.csv"
        bin_path2 = f"{bin_folder}\\{file2}.csv"

        shutil.move(raw_data_path1, bin_path1)
        shutil.move(raw_data_path2, bin_path2)

    ################## Step 8 - Storing the Data ########################################

    # Create SQL Tables.
    # TODO: Set it to false so it doesnt remove the existing database in actual tests. This is for testing purposes.
    database_setup(remove_existing=False)

    # Store results in database.
    store_results_sql(dfs_result)

    ################## Step 9 - It's done! ##################################


if __name__ == "__main__":
    # Setup Code Runner to allow user input:
    # https://stackoverflow.com/questions/50689210/how-to-setup-code-runner-in-visual-studio-code-for-python

    # https://www.youtube.com/watch?v=k27MJJLJNT4 Maybe try holoview. It allows you to easily overlay a line.

    # For plotting:
    # https://matplotlib.org/gallery/index.html#our-favorite-recipes

    main()
