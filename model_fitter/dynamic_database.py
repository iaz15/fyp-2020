from models import (Lubricant, VariableExperimentalConditions, StaticExperimentalConditions, Experiment, FrictionModelParametersLiquid)
from sqlalchemy import (Table, Column, Integer, Numeric, String, ForeignKey, Boolean, Float, MetaData, DateTime)
from sqlalchemy import func
from sqlalchemy import select, insert, update
from sqlalchemy import create_engine
import inspect

from sqlalchemy.schema import DropTable, CreateTable
from sqlalchemy.orm import scoped_session, sessionmaker

from numba.experimental import jitclass
from numba import int32, float32, double, typed, typeof, boolean

import os
import re

from contextlib import contextmanager

def get_all_properties(object_) -> list:
    """ Get all properties in the same order they are defined """
    return list(object_.__dict__.keys())

def camel_to_snake(name):
    """ Converts CamelCase to snake_case """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

spec = [
    ('lambda_1', float32),
    ('lambda_2', float32),
    ('c', float32),
    ('k_1', float32),
    ('k_2', float32),
    ('k_3', float32),
    ('k_4', float32),
    ('k_5', float32),
    ('D', float32),
    ('k_alpha', float32),
    ('Q_alpha', float32),
    ('K', float32),
    ('Q_K', float32),
]

# @jitclass(spec)
class FrictionModelParametersLiquidSolid():

    def __init__(self, lambda_1, lambda_2, c, k_1, k_2, k_3, k_4, k_5, D, k_alpha, Q_alpha, K, Q_K) -> None:

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.c = c
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.k_5 = k_5
        self.D = D
        self.k_alpha = k_alpha
        self.Q_alpha = Q_alpha
        self.K = K
        self.Q_K = Q_K

if __name__ == "__main__":
    """ 
    Work in Progress (TBC when time permits). 
    The aim is to be able to dynamically generate the schema for databases for fitting different models based on the equations used. 
    This will allow the use of the program for any experimental model (Not limited to friction), improving workflow efficiency.
    """

    parameters_2 = FrictionModelParametersLiquidSolid(lambda_1=20, lambda_2=0.8,
        c=0.02, k_1=0.75, k_2=2.0, D=2.0, Q_alpha=5.0, k_alpha=1.0,
        k_3=2.0, k_4=1.0, K=0.5, Q_K=7.5, k_5=1.0)

    ans = get_all_properties(parameters_2)
    print(ans)

    exit()
    # 1. User defines the classes.
    # 2. User creates objects from the classes.
    # 3. Through inspecting the objects, the databases are created!

    # User creates these classes to make a model
    lubricant_1 = Lubricant(1, 'ZEPF', True, False, density_liquid=1.125)               # Extra table which may not necessarily be there
    variable_condition_1 = VariableExperimentalConditions(450, 8, 0.34, 50, 25, 0)      # Mandatory columns - variable conditions (conditions)
    static_condition_1 = StaticExperimentalConditions(lubricant_1, 'P20', 0.3,          # Mandatory columns - static conditions (optimisation_groups)
        'AA7075', 0.7, 2, 'AlCrN', 0.2, 0.8)
    experiment_1 = Experiment(static_condition_1, variable_condition_1)                 # NOT REQUIRED (experiments)
    parameters_1 = FrictionModelParametersLiquid(mu0_lubricated = 1.69073,              # Mandatory columns - parameters (table name taken from class name)
        Q_lubricated = 9141.50683, mu0_dry = 10.94225, Q_dry = 9368.85126,
        eta_0 = 0.12, Q_eta = 11930, lambda_1 = 40.70, lambda_2 = 1.55, c = 0.00847,
        k_1 = 1.52, k_2 = 2.67, k_3 = 4.58)

    ##########################################
    ####### Dynamic creation of tables #######
    ##########################################

    # Try to delete existing table (for test purposes)
    try:
        os.remove("friction_model.db")
    except:
        print("Removal of database failed")

    print(camel_to_snake(parameters_1.__class__.__name__))

    # User provides this
    variable_properties = get_all_properties(experiment_1.variable_conditions)          # Get all variable properties
    static_properties = get_all_properties(experiment_1.static_conditions)              # Get all static properties

    engine = create_engine('sqlite:///friction_model.db')
    metadata = MetaData()

    columns_experiments = [
        Column('experiment_id', Integer(), primary_key=True, autoincrement=True),
        Column('condition_id', Integer(), ForeignKey('conditions.condition_id')),
        Column('filename', String(100)),
        Column('duplicate', Integer()),
        Column('created_on', DateTime(timezone=True), server_default=func.now()),
        Column('select', Boolean(), server_default='1'),
        Column('comments', String(100))
    ]

    ####################################################################
    ############# Create main table number 1 (experiments) #############
    ####################################################################

    Table('experiments', metadata, *columns_experiments)

    # Preset column at the start
    columns_variable_conditions = []
    columns_variable_conditions.append(Column('condition_id', Integer(), primary_key=True, autoincrement=True))
    columns_variable_conditions.append(Column('group_id', Integer(), ForeignKey('optimisation_groups.group_id')))

    for variable_condition_name in variable_properties:

        variable_condition_val = getattr(variable_condition_1, variable_condition_name)

        if isinstance(variable_condition_val, int):
            columns_variable_conditions.append(Column(str(variable_condition_name), Integer()))
        elif isinstance(variable_condition_val, str):
            columns_variable_conditions.append(Column(str(variable_condition_name), String(55)))
        elif isinstance(variable_condition_val, float):
            columns_variable_conditions.append(Column(str(variable_condition_name), Float()))
        elif isinstance(variable_condition_val, bool):
            columns_variable_conditions.append(Column(str(variable_condition_name), Boolean()))
        else:
            # Otherwise, its probably an object, then it should be an id columns
            # From the name, you get Lubricant, then .lower() -> lubricant -> add _id -> lubricant_id
            table_name = re.findall(r"'(.*?)'", type(variable_condition_val).__str__(type(variable_condition_val)))[0]

            # If it is imported there will be a "." in the name
            if "." in table_name:
                table_name = table_name.split(".")[1].lower()

            columns_variable_conditions.append(Column(f"{table_name}_id", Integer(), ForeignKey(f"{table_name}.{table_name}_id")))

            # If there is a class that means an extra table is required - Insert the specs in this table
            extra_table_properties = get_all_properties(variable_condition_val)

            columns_extra_table = []
            for extra_table_property in extra_table_properties:
                print(extra_table_property)
                if 'id' in extra_table_property:
                    columns_extra_table.append(Column(f"{table_name}_id", Integer(), primary_key=True, autoincrement=True))
                elif isinstance(extra_table_property, int):
                    columns_extra_table.append(Column(extra_table_property, Integer()))
                elif isinstance(extra_table_property, str):
                    columns_extra_table.append(Column(extra_table_property, String(55)))
                elif isinstance(extra_table_property, float):
                    columns_extra_table.append(Column(extra_table_property, Float()))
                elif isinstance(extra_table_property, bool):
                    columns_extra_table.append(Column(extra_table_property, Boolean()))

            # Sub table 1, 2, 3 ...
            Table(table_name, metadata, *columns_extra_table)

    #########################################################################
    ################ Create main table number 2 (conditions) ################
    #########################################################################

    # Preset column at the end
    columns_variable_conditions.append(Column('avg_filename', String(100)))
    Table('conditions', metadata, *columns_variable_conditions)

    ### OPTIMISATION GROUPS TABLE
    columns_static_conditions = []
    # Preset columns at the start
    columns_static_conditions.append(Column('group_id', Integer(), primary_key=True, autoincrement=True))

    for static_condition_name in static_properties:

        static_condition_val = getattr(static_condition_1, static_condition_name)

        if isinstance(static_condition_val, int):
            columns_static_conditions.append(Column(str(static_condition_name), Integer()))
        elif isinstance(static_condition_val, str):
            columns_static_conditions.append(Column(str(static_condition_name), String(55)))
        elif isinstance(static_condition_val, float):
            columns_static_conditions.append(Column(str(static_condition_name), Float()))
        elif isinstance(static_condition_val, bool):
            columns_static_conditions.append(Column(str(static_condition_name), Boolean()))
        else:
            # Otherwise, its probably an object, then it should be an id columns
            # From the name, you get Lubricant, then .lower() -> lubricant -> add _id -> lubricant_id
            table_name = re.findall(r"'(.*?)'", type(static_condition_val).__str__(type(static_condition_val)))[0]

            # If it is imported there will be a "." in the name
            if "." in table_name:
                table_name = table_name.split(".")[1].lower()

            columns_static_conditions.append(Column(f"{table_name}_id", Integer(), ForeignKey(f"{table_name}.{table_name}_id")))

            # If there is a class that means an extra table is required - Insert the specs in this table
            extra_table_properties = get_all_properties(static_condition_val)

            columns_extra_table = []
            for extra_table_property in extra_table_properties:
                if 'id' in extra_table_property:
                    columns_extra_table.append(Column(f"{table_name}_id", Integer(), primary_key=True, autoincrement=True))
                elif isinstance(extra_table_property, int):
                    columns_extra_table.append(Column(extra_table_property, Integer()))
                elif isinstance(extra_table_property, str):
                    columns_extra_table.append(Column(extra_table_property, String(55)))
                elif isinstance(extra_table_property, float):
                    columns_extra_table.append(Column(extra_table_property, Float()))
                elif isinstance(extra_table_property, bool):
                    columns_extra_table.append(Column(extra_table_property, Boolean()))

            # Sub table 1, 2, 3 ...
            Table(table_name, metadata, *columns_extra_table)

    ############################################################################
    ############# Create main table number 3 (optimisation_groups) #############
    ############################################################################

    Table('optimisation_groups', metadata, *columns_static_conditions)

    # ### PARAMETERS TABLE
    # COLUMNS_PARAMETERS = []
    # # Name is determined by class name

    metadata.create_all(engine)

    ############# Create main table number 4 (parameters) - Assume only 1. If more, create more of the same #############

    # Then we know that there should be another table

    # https://gist.github.com/djrobstep/998b9779d0bbcddacfef5d76a3d0921a

    # TABLE 1: experiments

    # TABLE 2: conditions

    # TABLE 3: optimisation_groups / groups

    # TABLE 4: lubricant
    # (Something in the table which is a class - it belongs to another table with a FOREIGN KEY)
    # It should have a column called id
    # The table name is what is in between the ' ' and split by the . if there is any
    # Example: 'models.Lubricant' -> Lubricant table. The id name will be '<table_name>_id'
    # The first column will be lubricant_id. It will be a foreign key to the previous table
    # In the previous table, the colum will be lubricant_id

    # TABLE 5: model_parameters_(liquid/solid/...)


    # conditions = Table('conditions', metadata,
    #     Column('condition_id', Integer(), primary_key=True, autoincrement=True),
    #     Column('group_id', Integer(), ForeignKey('optimisation_groups.group_id')),
    #     Column('temperature', Integer()),
    #     Column('force', Integer()),
    #     Column('pressure', Float()),
    #     Column('speed', Integer()),
    #     Column('lubricant_thickness', Integer()),
    #     Column('avg_filename', String(100)),
    # )