import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

from datetime import datetime
from sqlalchemy import create_engine, MetaData, DateTime
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import (Table, Column, Integer, Numeric, String, ForeignKey, Boolean, Float)
from sqlalchemy import func
from sqlalchemy import select, insert, update

from rename_files import Experiment
from data_averaging import average_datasets
import data_fitting

import os
import sys
import shutil
import numpy as np

class DataFrameHolder:
    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return "<DataFrameHolder Object>"


def extract_data(filename, directory=None):

    if directory!=None:
        filename_path = os.path.join(directory, filename)
    else:
        filename_path = filename

    df = pd.read_csv(filename_path)
    df_holder = DataFrameHolder(df)

    return df_holder


def create_database():
    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = MetaData()

    experiments = Table('experiments', metadata,
        Column('experiment_id', Integer(), primary_key=True, autoincrement=True),
        Column('condition_id', Integer(), ForeignKey('conditions.condition_id')),
        Column('filename', String(100)),
        Column('duplicate', Integer()),
        Column('created_on', DateTime(timezone=True), server_default=func.now()),
        Column('select', Boolean(), server_default='1'),
        Column('comments', String(100))
    )

    conditions = Table('conditions', metadata,
        Column('condition_id', Integer(), primary_key=True, autoincrement=True),
        Column('group_id', Integer(), ForeignKey('condition_groups.group_id')),
        Column('temperature', Integer()),
        Column('force', Integer()),
        Column('pressure', Float()),
        Column('speed', Integer()),
        Column('lubricant_thickness', Integer()),
        Column('equiv_solid_thickness', Float(), index=True),
        Column('avg_filename', String(100)),
    )

    condition_groups = Table('condition_groups', metadata,
        Column('group_id', Integer(), primary_key=True, autoincrement=True),
        Column('lubricant_id', Integer(), ForeignKey('lubricants.lubricant_id')),
        Column('pin_material', String(55)),
        Column('pin_roughness', Float()),
        Column('blank_material', String(55)),
        Column('blank_roughness', Float()),
        Column('blank_thickness', Float()),
        Column('coating_material', String(55)),
        Column('coating_roughness', Float()),
        Column('coating_thickness', Float()),
    )

    model_params_liquid = Table('model_params_liquid', metadata,
        Column('entry_id', Integer(), primary_key=True, autoincrement=True),
        Column('group_id', Integer(), ForeignKey('condition_groups.group_id')),

        Column('eta_0', Float()),
        Column('Q_eta', Float()),
        Column('mu0_lubricated', Float()),
        Column('Q_lubricated', Float()),
        Column('mu0_dry', Float()),
        Column('Q_dry', Float()),

        Column('lambda_1', Float()),
        Column('lambda_2', Float()),
        Column('c', Float()),
        Column('k_1', Float()),
        Column('k_2', Float()),
        Column('k_3', Float())
    )

    model_params_liquid_solid = Table('model_params_liquid_solid', metadata,
        Column('entry_id', Integer(), primary_key=True, autoincrement=True),
        Column('group_id', Integer(), ForeignKey('condition_groups.group_id')),

        Column('eta_0', Float()),
        Column('Q_eta', Float()),
        Column('mu0_lubricated', Float()),
        Column('Q_lubricated', Float()),
        Column('mu0_dry', Float()),
        Column('Q_dry', Float()),

        Column('lambda_1', Float()),
        Column('lambda_2', Float()),
        Column('c', Float()),
        Column('k_1', Float()),
        Column('k_2', Float()),
        Column('k_3', Float())
    )

    lubricants = Table('lubricants', metadata,
        Column('lubricant_id', Integer(), primary_key=True, autoincrement=True),
        Column('name', String(55), index=True),
        Column('includes_liquid', Boolean(), index=True),
        Column('includes_solid', Boolean(), index=True),
        Column('density_liquid', Float(), index=True),
        Column('avg_hardness_solid', Float(), index=True)
    )

    intermediate_values = Table('intermediate_values', metadata,
        Column('intermediate_value_id', Integer(), primary_key=True, autoincrement=True),
        Column('group_id', Integer()),
        Column('temperature', Integer()),
        Column('coefficient_of_friction_dry', Float()),
        Column('coefficient_of_friction_lubricated', Float()),
        Column('kinematic_viscosity', Float()),
        Column('select', Boolean(), server_default='1')
    )

    metadata.create_all(engine)
    con.close()


def insert_example_lubricants():

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    lubricants = sqlalchemy.Table('lubricants', metadata, autoload=True, autoload_with=engine)

    values_list = [
        {'lubricant_id': 1, 'name': 'Omega 35'},
        {'lubricant_id': 2, 'name': 'ZEPF'}
    ]

    stmt = insert(lubricants)

    results = con.execute(stmt, values_list)

    con.close()


def add_experiment_database(experiment):

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    experiments = sqlalchemy.Table('experiments', metadata, autoload=True, autoload_with=engine)
    conditions = sqlalchemy.Table('conditions', metadata, autoload=True, autoload_with=engine)
    condition_groups = sqlalchemy.Table('condition_groups', metadata, autoload=True, autoload_with=engine)
    lubricants = sqlalchemy.Table('lubricants', metadata, autoload=True, autoload_with=engine)

    # Add to experiment values into the right columns and tables

    ### First determine group_id & insert row as necessary
    query = select([condition_groups.c.group_id])\
                    .where(getattr(condition_groups.c, 'lubricant_id') == getattr(experiment.conditions, 'lubricant_id'))\
                    .where(getattr(condition_groups.c, 'blank_material') == getattr(experiment.conditions, 'blank_material'))\
                    .where(getattr(condition_groups.c, 'blank_roughness') == getattr(experiment.conditions, 'blank_roughness'))\
                    .where(getattr(condition_groups.c, 'blank_material') == getattr(experiment.conditions, 'blank_material'))\
                    .where(getattr(condition_groups.c, 'pin_material') == getattr(experiment.conditions, 'pin_material'))\
                    .where(getattr(condition_groups.c, 'pin_roughness') == getattr(experiment.conditions, 'pin_roughness'))\
                    .where(getattr(condition_groups.c, 'coating_material') == getattr(experiment.conditions, 'coating_material'))\
                    .where(getattr(condition_groups.c, 'coating_roughness') == getattr(experiment.conditions, 'coating_roughness'))\
                    .where(getattr(condition_groups.c, 'coating_thickness') == getattr(experiment.conditions, 'coating_thickness'))

    # There should only 1 matching group_id maximum
    result_proxy  = con.execute(query)
    matching_group_num = result_proxy.scalar()

    if matching_group_num:
        # If there is an existing match set group_id to that
        experiment.set_group_id(matching_group_num)
    else:
        # Otherwise insert as a new row
        query = select([func.max(condition_groups.c.group_id)])
        result_proxy  = con.execute(query)
        max_group_id = result_proxy.scalar()

        if max_group_id is None:
            max_group_id = 0

        experiment.set_group_id(max_group_id + 1)

        insert_stmt = insert(condition_groups).\
                        values(lubricant_id=experiment.conditions.lubricant_id, blank_material=experiment.conditions.blank_material,\
                                blank_roughness=experiment.conditions.blank_roughness, blank_thickness=experiment.conditions.blank_thickness,\
                                pin_material=experiment.conditions.pin_material, pin_roughness=experiment.conditions.pin_roughness,\
                                coating_material=experiment.conditions.coating_material, coating_roughness=experiment.conditions.coating_roughness,\
                                coating_thickness=experiment.conditions.coating_thickness)

        results = con.execute(insert_stmt)

    ### Then determine condition_id & insert row as necessary
    query = select([conditions.c.condition_id])\
                .where(conditions.c.group_id == experiment.group_id)\
                .where(conditions.c.temperature == experiment.conditions.temperature)\
                .where(conditions.c.speed == experiment.conditions.speed)\
                .where(conditions.c.force == experiment.conditions.force)\
                .where(conditions.c.pressure == experiment.conditions.pressure)\
                .where(conditions.c.lubricant_thickness == experiment.conditions.lubricant_thickness)\

    # There should only 1 matching condition_id maximum
    result_proxy  = con.execute(query)
    matching_condition_num = result_proxy.scalar()

    if matching_condition_num:
        # If there is an existing match set condition_id to that
        experiment.set_condition_id(matching_condition_num)
    else:
        # Otherwise insert as a new row
        query = select([func.max(conditions.c.condition_id)])
        result_proxy  = con.execute(query)
        max_condition_id = result_proxy.scalar()

        if max_condition_id is None:
            max_condition_id = 0

        experiment.set_condition_id(max_condition_id + 1)

        insert_stmt = insert(conditions).\
                        values(group_id=experiment.group_id, temperature=experiment.conditions.temperature,
                            speed=experiment.conditions.speed, force=experiment.conditions.force,
                            pressure=experiment.conditions.pressure, lubricant_thickness=experiment.conditions.lubricant_thickness)

        results = con.execute(insert_stmt)

    ### Then add the results to experiments table & determine duplicate id as necessary
    query = select([func.count(experiments.c.condition_id)]).where(experiments.c.condition_id == experiment.condition_id)

    duplicate_count  = con.execute(query).scalar()

    experiment.set_duplicate_number(duplicate_count)

    insert_stmt = insert(experiments).\
                    values(condition_id=experiment.condition_id,
                            filename=experiment.filename,
                            duplicate=experiment.duplicate_number)

    results = con.execute(insert_stmt)

    con.close()


def initialise_new_database():
    try:
        # remove existing database
        os.remove('friction_model.db')
    except:
        pass

    # create the database
    create_database()

    # insert sample lubricants that will be used
    insert_example_lubricants()

    ### 1. USER UPLOADS DATA AND INPUTS TEST CONDITIONS
    # create sample experiment objects that will be passed in to the database
    experiments_dict = create_sample_datasets()

    # insert each experiment into the database and populate columns accordingly
    for experiment in experiments_dict.values():
        add_experiment_database(experiment)


def create_sample_datasets():

    # Try 5N, 80mm/s and 350 deg C

    # 5N: 0.34 MPa, 100 mm/s: 0.34 MPa, 200 degrees celcius: 0.41 MPa, 8N: 0.43 MPa

    # Example of objects the user will create. This will be stored in a dictionary before adding to the
    # database
    experiments_dict = {}
    experiments_dict[1] = Experiment(1, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 50, 5, 0.34, 25)
    experiments_dict[2] = Experiment(2, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 50, 5, 0.34, 25)
    experiments_dict[3] = Experiment(3, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 50, 5, 0.34, 25)
    experiments_dict[4] = Experiment(4, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 100, 5, 0.34, 25)
    experiments_dict[5] = Experiment(5, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 100, 5, 0.34, 25)
    experiments_dict[6] = Experiment(6, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 100, 5, 0.34, 25)
    experiments_dict[7] = Experiment(7, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 200, 50, 5, 0.41, 25)
    experiments_dict[8] = Experiment(8, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 200, 50, 5, 0.41, 25)
    experiments_dict[9] = Experiment(9, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 200, 50, 5, 0.41, 25)
    experiments_dict[10] = Experiment(10, 1, 'P20', 0.8, 'AA7075', 0.5, 5, 'None', 0, 0, 250, 8, 50, 0.43, 25)

    experiments_dict[1].add_output_filename('1_output.csv')
    experiments_dict[2].add_output_filename('2_output.csv')
    experiments_dict[3].add_output_filename('3_output.csv')
    experiments_dict[4].add_output_filename('4_output.csv')
    experiments_dict[5].add_output_filename('5_output.csv')
    experiments_dict[6].add_output_filename('6_output.csv')
    experiments_dict[7].add_output_filename('7_output.csv')
    experiments_dict[8].add_output_filename('8_output.csv')
    experiments_dict[9].add_output_filename('9_output.csv')
    experiments_dict[10].add_output_filename('10_output.csv')

    return experiments_dict


def average_data(condition_id, plot_results=True, print_experiments=False):
    print(f"Averaging using condition id: {condition_id}")
    results_folder = 'results_friction'
    averaged_folder = 'averaged_friction'

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    experiments = sqlalchemy.Table('experiments', metadata, autoload=True, autoload_with=engine)
    conditions = sqlalchemy.Table('conditions', metadata, autoload=True, autoload_with=engine)
    condition_groups = sqlalchemy.Table('condition_groups', metadata, autoload=True, autoload_with=engine)
    lubricants = sqlalchemy.Table('lubricants', metadata, autoload=True, autoload_with=engine)

    stmt = select([experiments.c.experiment_id, conditions.c.condition_id, experiments.c.filename, conditions.c.temperature, conditions.c.speed,
                   conditions.c.force, conditions.c.lubricant_thickness, experiments.c.select])
    stmt = stmt.select_from(experiments.join(conditions))
    stmt = stmt.where(conditions.c.condition_id==condition_id)\
               .where(experiments.c.select==1)

    result_proxy = con.execute(stmt)
    results = result_proxy.fetchall()

    df = pd.DataFrame(results)
    df.columns = result_proxy.keys()

    if print_experiments==True:
        print(df)

    dfs = []
    for filename in df.filename:
        dfs.append(pd.read_csv(os.path.join(results_folder, filename)))

    df_averaged_data = average_datasets(dfs, plot_results=plot_results)

    if isinstance(df_averaged_data, int):
        # If there is only 1 set of data it will return a 0 (an int)
        con.close()
        return 0
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


def add_fitting_params_db(group_id, params):

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    model_params_liquid = sqlalchemy.Table('model_params_liquid', metadata, autoload=True, autoload_with=engine)

    # Update database with averaged file name
    u = update(model_params_liquid).where(model_params_liquid.c.group_id == int(group_id))
    u = u.values(eta_0=params['eta_0'], Q_eta=params['Q_eta'], mu0_lubricated=params['mu0_lubricated'],
                 Q_lubricated=params['Q_lubricated'], mu0_dry=params['mu0_dry'], Q_dry=params['Q_dry'],
                 lambda_1=params['lambda_1'], lambda_2=params['lambda_2'], c=params['c'],
                 k_1=params['k_1'], k_2=params['k_2'], k_3=params['k_3'])

    result = con.execute(u)

    con.close()


def extract_data_fitting(group_id):
    df_conditions_table = pd.DataFrame()
    params_conditions_dict = {}

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    conditions = sqlalchemy.Table('conditions', metadata, autoload=True, autoload_with=engine)
    condition_groups = sqlalchemy.Table('condition_groups', metadata, autoload=True, autoload_with=engine)

    # We want to get the (model params) + (temperature + speed + force + lubricant thickness) + (avg_filenames)

    ### 1. Get the variable test conditions for each experiment for a group
    stmt = select([conditions.c.condition_id, conditions.c.group_id, conditions.c.avg_filename,
                   conditions.c.temperature, conditions.c.speed, conditions.c.force, conditions.c.pressure, conditions.c.lubricant_thickness])
    stmt = stmt.where(conditions.c.group_id==group_id)\
               .where(conditions.c.avg_filename!=None)

    result_proxy = con.execute(stmt)
    results = result_proxy.fetchall()

    ### 2. Obtain a list of dictionaries for the variable test conditions
    df_conditions_table = pd.DataFrame(results)
    df_conditions_table.columns = result_proxy.keys()

    ### 3. Get the params and static conditions for a group
    stmt = select([condition_groups])
    stmt = stmt.where(condition_groups.c.group_id==group_id)

    result_proxy = con.execute(stmt)
    results = result_proxy.fetchall()

    ### 4. Obtain a dictionary for params + static test condition
    # Unpack the values into a single list
    results_unpacked = []
    for result in results:
        results_unpacked = [item for sublist in results for item in sublist]

    params_conditions_dict = dict(zip(result_proxy.keys(), results_unpacked))

    con.close()
    return df_conditions_table, params_conditions_dict


def transform_params_fitting(df_conditions_table, params_conditions_dict):
    df_variable_conditions = df_conditions_table.loc[:, ['temperature', 'speed', 'force', 'lubricant_thickness', 'pressure']]
    dict_variable_conditions = df_variable_conditions.to_dict('records')
    testing_params_all = [{**record, **params_conditions_dict} for record in dict_variable_conditions]

    return testing_params_all


def extract_experimental_data(df_conditions_table):
    sd_measured_set = []
    cof_measured_set = []
    for index, row in df_conditions_table.iterrows():
        data = pd.read_csv(os.path.join(averaged_data_dir, row['avg_filename']))
        sd_measured_set.append(data['sliding_distance_(mm)'].values)
        cof_measured_set.append(data['coefficient_of_friction'].values)

    # Convert to numpy array to make it compatible
    sd_measured_set = np.array(sd_measured_set, dtype=object)
    cof_measured_set = np.array(cof_measured_set, dtype=object)

    return sd_measured_set, cof_measured_set


def update_model_params(group_id, params):
    """ Using the name of the lubricant, the correct row will be updated. """

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    condition_groups = sqlalchemy.Table('condition_groups', metadata, autoload=True, autoload_with=engine)

    print("Group id", group_id)
    # Update database with averaged file name
    u = update(condition_groups).where(condition_groups.c.group_id == int(group_id))
    u = u.values(lambda_1=params['lambda_1'], lambda_2=params['lambda_2'], c=params['c'],
                 k_1=params['k_1'], k_2=params['k_2'], k_3=params['k_3'])

    con.execute(u)
    con.close()


if __name__ == "__main__":
    averaged_data_dir = 'averaged_friction'

    ### 1. INITIALISE THE DATABASE WITH STARTING VALUES (Removes existing)
    initialise_new_database()

    # ### 2. USER WILL WANT TO DO SOME DATA PROCESSING ON REPEATED FILES - Modify condition_id to see results
    # # Use information in database and data in stored files to average the data accordingly
    # # There are 4 conditions. Change the value to see the effect
    # condition_id = 1
    # average_data(1, plot_results=False, print_experiments=False)
    # average_data(2, plot_results=False, print_experiments=False)
    # average_data(3, plot_results=False, print_experiments=False)


    # ### 3. POPULATE THE PARAMETER COLUMNS WITH INITIAL ESTIMATES FOR THAT GROUP
    # # High speed Omega35
    # group_id = 1
    # group_1_params = {'eta_0': 0.13, 'Q_eta': 11930, 'mu0_lubricated': 1.69, 'Q_lubricated': 9141.5,
    #                       'mu0_dry': 10.94, 'Q_dry': 9368.8, 'lambda_1': 40, 'lambda_2': 1.5, 'c': 0.00847,
    #                       'k_1': 1.52, 'k_2': 2.67, 'k_3': 4.58}

    # add_fitting_params_db(group_id=group_id, params=group_1_params)

    # ### 4. Extract the files and information necessary for plotting and transform into the correct format for manual fitting
    # group_id = 1
    # df_conditions_table, params_conditions_dict = extract_data_fitting(group_id=group_id)
    # testing_params_all = transform_params_fitting(df_conditions_table, params_conditions_dict)
    # sd_measured_set, cof_measured_set = extract_experimental_data(df_conditions_table)

    # # ### 5. Plot results
    # # list_zipped_results = list(zip(testing_params_all, sd_measured_set, cof_measured_set))
    # # plotting_range = np.linspace(0, 88, 420)
    # # base = testing_params_all[0]            # Make sure the selected one is the right set of params
    # # data_fitting.plot_graphs(plotting_range=plotting_range, base=base, list_zipped_results=list_zipped_results, time_input=False, title="Interactive Friction Model")

    # # ### 6. Manual fitting
    # data_fitting.manual_fitting_slider(testing_params_all, sd_measured_set, cof_measured_set)

    # # ### 7. Extract the files and information necessary for plotting and transform into the correct format for automatic fitting
    # # group_id = 1
    # # df_conditions_table, params_conditions_dict = extract_data_fitting(group_id=group_id)
    # # testing_params_all = transform_params_fitting(df_conditions_table, params_conditions_dict)
    # # sd_measured_set, cof_measured_set = extract_experimental_data(df_conditions_table)

    # # ### 8. Automatic fitting
    # # RANGE_C = [0.005, 0.02]
    # # RANGE_K1 = [1, 3]
    # # RANGE_K2 = [2, 3]
    # # RANGE_K3 = [2, 10]
    # # RANGE_LAMBDA1 = [0.1, 42]
    # # RANGE_LAMBDA2 = [0.1, 3]

    # # params_solved = data_fitting.optimise_friction_results(testing_params_all, sd_measured_set, cof_measured_set,
    # #                                                        RANGE_C, RANGE_K1, RANGE_K2, RANGE_K3, RANGE_LAMBDA1, RANGE_LAMBDA2,
    # #                                                        time_input=False, plot_results=True)
    # # lubricant_id = params_solved[0]['lubricant_id']
    # # group_id = params_solved[0]['group_id']
    # # blank_material = params_solved[0]['blank_material']
    # # pin_material = params_solved[0]['pin_material']

    # # print(params_solved)
    # # while True:
    # #     print("Do you want to overwrite the old parameter values? (Y/N)")
    # #     print(f"Group ID: {group_id}")
    # #     print(f"Lubricant ID: {lubricant_id}")
    # #     print(f"Blank Material: {blank_material}")
    # #     print(f"Pin Material: {pin_material}")
    # #     option = input("")

    # #     if option == 'Y':
    # #         update_model_params(params_solved[0]['group_id'], params_solved[0])
    # #         break
    # #     elif option == 'N':
    # #         break
    # #     else:
    #         # pass
