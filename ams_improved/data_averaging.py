import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import (Table, Column, Integer, Numeric, String, ForeignKey, Float,
                        PrimaryKeyConstraint, UniqueConstraint, CheckConstraint)
from sqlalchemy import insert, select
from sqlalchemy import func
from sqlalchemy import update
from pathlib import Path
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
        print("")
        return 0

    # For each of them, make 'time_elapsed_(s)' the index before merging.
    for experimental_dataset in experimental_datasets:
        # There may be duplicates time_elapsed_(s) for some reason. These need to be removed for concatenation
        #   This has been taken care of at an earlier stage in data processing
        experimental_dataset.set_index('time_elapsed_(s)', inplace=True)

    # Merge the data, then interpolate in between.
    # Drop null values (Typically at the ends)
    # https://stackoverflow.com/questions/35084071/concat-dataframe-reindexing-only-valid-with-uniquely-valued-index-objects
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

    print("")

    return averaged_data

def average_data(condition_id, plot_results=True, print_experiments=False):
    print(f"Averaging using condition id: {condition_id}")
    results_folder = 'results_friction'
    averaged_folder = 'averaged_friction'

    if not Path('friction_model.db').is_file():
        raise Exception("Cannot use average_data function without an existing database")

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    experiments = sqlalchemy.Table('experiments', metadata, autoload=True, autoload_with=engine)
    conditions = sqlalchemy.Table('conditions', metadata, autoload=True, autoload_with=engine)
    condition_groups = sqlalchemy.Table('condition_groups', metadata, autoload=True, autoload_with=engine)
    lubricants = sqlalchemy.Table('lubricants', metadata, autoload=True, autoload_with=engine)

    stmt = select([experiments.c.experiment_id, conditions.c.condition_id, experiments.c.filename, conditions.c.temperature_degC, conditions.c.speed_mmpersecond,
                   conditions.c.force_N, conditions.c.lubricant_thickness_micrometres, experiments.c.select])
    stmt = stmt.select_from(experiments.join(conditions))
    stmt = stmt.where(conditions.c.condition_id==condition_id)\
               .where(experiments.c.select==1)

    result_proxy = con.execute(stmt)
    results = result_proxy.fetchall()

    if len(results) == 0:
        print(f"Condition with id {condition_id} does not exist, aborting.")
        con.close()
        return None

    df = pd.DataFrame(results)
    df.columns = result_proxy.keys()

    if print_experiments==True:
        print(df)

    dfs = []
    for filename in df.filename:
        dfs.append(pd.read_csv(os.path.join(results_folder, filename)))

    experiment_num_list = df['experiment_id'].to_list()
    experiment_num_string = '_'.join([str(num) for num in experiment_num_list])

    print(f"Using experiments: {experiment_num_list}")

    df_averaged_data = average_datasets(dfs, plot_results=plot_results)

    if isinstance(df_averaged_data, int):
        # If there is only 1 set of data it will return a 0 (an int)
        con.close()
        return None
    else:
        # Create the folder if it doesn't exist
        if not os.path.exists(averaged_folder):
            os.makedirs(averaged_folder)
        else:
            pass

        condition_id = df.condition_id.iloc[0]

        # save result to averaged folder
        averaged_filename = f'C{condition_id}_averaged.csv'

        df_averaged_data.to_csv(os.path.join(averaged_folder, averaged_filename), sep=',', index=False)

        # Update database with averaged file name
        u = update(conditions).where(conditions.c.condition_id == int(condition_id))
        u = u.values(avg_filename=averaged_filename)
        result = con.execute(u)

        con.close()

if __name__ == "__main__":
    average_data(1, plot_results=True, print_experiments=False)
    average_data(2, plot_results=True, print_experiments=False)
    average_data(3, plot_results=True, print_experiments=False)
    average_data(4, plot_results=True, print_experiments=False)
    average_data(5, plot_results=True, print_experiments=False)
    average_data(6, plot_results=True, print_experiments=False)
    average_data(7, plot_results=True, print_experiments=False)
    average_data(8, plot_results=True, print_experiments=False)