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
import math


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

    if len(experimental_datasets) == 1:
        print("Only one dataset used, skipping averaging")
        return 0

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

if __name__ == "__main__":
    pass