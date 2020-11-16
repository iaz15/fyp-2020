from lmfit.printfuncs import fit_report
import numpy as np
import scipy.io as sio
import math
import scipy
import time
import glob
import os
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from sklearn.linear_model import LinearRegression
import scipy.io
import lmfit
from scipy.io import loadmat
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.legend import Legend
import logging
from prettytable import PrettyTable
import sqlalchemy as db
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import (Table, Column, Integer, Numeric, String, ForeignKey, Float,
                        PrimaryKeyConstraint, UniqueConstraint, CheckConstraint)
from sqlalchemy import insert
from sqlalchemy import func
from sqlalchemy import update
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, AxesWidget
from numpy import pi, sin, cos


# Max line length for code: 78 chars
# Max line length for docstring: 72 chars
# Docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
def friction_coefficient_model(beta, paras):
    """
    The first equation of the interactive friction model.

    Parameters
    ----------
    beta: float
        The lubricated area ratio. This varies from 0 to 1.
    paras: dict
        A dictionary containing information about the test conditions,
        properties of the lubricant and material as well as model
        constants used in the interactive friction model.

    Returns
    -------
    float
        The coefficient of friction, mu given the test conditions.
    """
    mu_lubricated = arrhenius_solve_cof(paras['mu0_lubricated'], paras['Q_lubricated'], convert_to_kelvin(paras['temperature']))
    mu_dry = arrhenius_solve_cof(paras['mu0_dry'], paras['Q_dry'], convert_to_kelvin(paras['temperature']))

    mu = (1 - beta)*mu_lubricated + beta*mu_dry
    return mu


def contribution_ratio_model(h, paras):
    """
    The second equation of the interactive friction model.

    Parameters
    ----------
    h: float
        Lubricant thickness at a given time.
    paras: dict
        A dictionary containing information about the test conditions,
        properties of the lubricant and material as well as model
        constants used in the interactive friction model.

    Returns
    -------
    int
        The lubricated area ratio. This varies from 0 to 1.


    """
    lambda_1 = paras['lambda_1']
    lambda_2 = paras['lambda_2']

    beta = math.exp(-(lambda_1 * h)**lambda_2)
    return beta


def film_thickness_model(h, t, paras, idx=None, changing_inputs=None):
    """
    The third equation of the interactive friction model.
    Note that P depends on T and F. This must be obtained before carrying
    out optimisation. If changing_inputs is True that means that the
    input is changing. This is meant to be used with the odeint function.

    Parameters
    ----------
    h: float
        Lubricant thickness at a given time.
    t: float
        Time step at which the function is meant to evaluate at.
    paras:
        A dictionary containing information about the test
        conditions, properties of the lubricant and material as well
        as model constants used in the interactive friction model.
    idx: int, optional
        This is the integer
    changing_inputs: bool, optional
        This is to indicate if inputs (Temperature, Pressure) are
        changing. If so, processing will be different and the paras
        dictionary will contain a dictionary containing a pandas
        dataframe with entries containing their variation over time.

    Returns
    -------
    float:
        The rate of change of the lubricant thickness.
    """
    if changing_inputs == None or changing_inputs == False:
        # If changing_inputs is not specified, use constant P, T and v
        c = paras['c']
        k_1 = paras['k_1']
        k_2 = paras['k_2']
        k_3 = paras['k_3']

        P = paras['pressure']
        T = paras['temperature']

        v = paras['speed']

        eta_0 = paras['eta_0']
        Q_eta = paras['Q_eta']

        # Solve for eta using Temperature and Arrhenius constants
        eta = arrhenius_solve_eta(eta_0, Q_eta, convert_to_kelvin(T))

        h_rate = -h*(c*(P**k_1)*(v**k_2)/(eta**k_3))


    elif changing_inputs == True:
        # If changing_inputs is set to True that means P, T and v are changing
        # That means that there is a changing_values key in the dictionary

        # Unpack parameters
        h_t = h[0]

        T_t = float(paras['changing_values'].loc[idx, "T"])
        P_t = float(paras['changing_values'].loc[idx, "P"])

        # Velocity is always constant
        # v = float(paras['changing_values'].loc[idx, "v"])
        v = paras['speed']

        c = paras['c']
        k_1 = paras['k_1']
        k_2 = paras['k_2']
        k_3 = paras['k_3']

        eta_0 = paras['eta_0']
        Q_eta = paras['Q_eta']

        # Solve for eta using Temperature and Arrhenius constants
        eta = arrhenius_solve_eta(eta_0, Q_eta, convert_to_kelvin(T_t))

        h_rate = -h_t*(c*(P_t**k_1)*(v**k_2)/(eta**k_3))

    else:
        # If the changing_inputs is neither True or False, raise an exception
        raise Exception(f'time_input should be Boolean or None (T/F).\
                          The changing_inputs parameter was set to \
                          {changing_inputs}')

    return h_rate


def arrhenius_solve_cof(param_0, Q, T):
    """
    Given constants for the Arrhenius equation and the temperature
    (in Kelvin), gives the coefficient of friction at a given
    Temperature.

    Parameters
    ----------
    param_0: Union[int, float]
        The pre-exponential factor for the Arrhenius equation to solve
        for the coefficient of friction.
    Q: Union[int, float]
        Activation energy, in Joules/mole
    T: Union[int, float]
        Temperature, in Kelvin.

    Returns
    -------
    float
        The coefficient of friction at the given Temperature

    """
    R = 8.31
    cof = param_0 * np.exp(-Q/(R*T))
    return cof


def arrhenius_solve_eta(param_0, Q, T):
    """
    Given constants for arrhenius equation and the Temperature
    (in Kelvin), returns the value at that condition. This is
    for eta as it doesn't have a -ve

    Parameters
    ----------
    param_0: Union[int, float]
        The pre-exponential factor for the Arrhenius equation to solve
        for the coefficient of friction.
    Q: Union[int, float]
        Activation energy, in Joules/mole
    T: Union[int, float]
        Temperature, in Kelvin.

    Returns
    -------
    float
        The coefficient of friction at the given Temperature
    """
    R = 8.31
    ans = param_0 * np.exp(Q/(R*T))
    return ans


def convert_to_kelvin(T):
    """
    Converts temperature in Degrees Celcius to Kelvin

    Parameters
    ----------
    T: int
        Temperature in Degrees Celcius.

    Returns
    -------
    float
        Temperature converted from Degrees Celcius to Kelvin.
    """
    return T + 273.1


def print_evaluated_cof_eta(paras):
    """
    Evaluates the coefficients of friction given a dictionary of experimental parameters.

    Parameters
    ----------
    paras: dict
        A dictionary containing information about the test
        conditions, properties of the lubricant and material as well
        as model constants used in the interactive friction model.


    Returns
    -------
    None
        Just prints the evaluated eta and coefficient of friction
        (fully lubricated and dry) given the test conditions.
    """
    eta = arrhenius_solve_eta(paras['eta_0'], paras['Q_eta'], convert_to_kelvin(paras['temperature']))
    cof_lub = arrhenius_solve_cof(paras['mu0_lubricated'], paras['Q_lubricated'], convert_to_kelvin(paras['temperature']))
    cof_dry = arrhenius_solve_cof(paras['mu0_dry'], paras['Q_dry'], convert_to_kelvin(paras['temperature']))

    print(f"Evaluated eta: {eta}")
    print(f"Evaluated CoF lubricated: {cof_lub}")
    print(f"Evaluated CoF dry: {cof_dry}")
    return None


def solve_all(x_input, lubricant_thickness, paras, time_input):
    """
    Default required x_input value: An array of sliding distances.

    If time_input = False:
        - x_input is an array of sliding distance.
    If time_input = True:
        - x_input is an array of time elapsed since the start of the test.

    lubricant_thickness is the initial lubricant volume.

    Parameters
    ----------
    x_input: int
        The values of x to be used. This will either be sliding distances
        or time elapsed since the start of the test. The required
        values to be input will depend on the state of time_input.
    lubricant_thickness: Union[int, float]
        The initial lubricant thickness in micrometres.
    paras: dict
        A dictionary containing information about the test
        conditions, properties of the lubricant and material as well
        as model constants used in the interactive friction model.
    time_input: bool
        True or False indicating whether the x_input data is time or
        sliding distance data.

    Returns
    -------
    ndarray
        1D numpy array containing the coefficient of friction at
        different times.

    Raises
    ------
    Exception
        If the time_input input is not a boolean.
    """

    if time_input == False:
        t = x_input/paras['speed']
    elif time_input == True:
        t = x_input
    else:
        raise Exception(f'time_input should be Boolean (T/F). The time_input was set to {time_input}')

    h_array = odeint(film_thickness_model, lubricant_thickness, t, args=(paras,))

    # If the graph breaks, adjust the decimals this is rounded to.
    h_array = np.around(h_array, decimals=9)
    # beta_array = np.apply_along_axis(contribution_ratio_model, 0, h_array, paras)
    # mu_array = np.apply_along_axis(friction_coefficient_model, 0, beta_array, paras)
    beta_array = np.array([contribution_ratio_model(h_i, paras) for h_i in h_array])
    mu_array = np.array([friction_coefficient_model(beta_i, paras) for beta_i in beta_array])

    return mu_array


def model_dataset(params, i, x_input, time_input):
    """
    Calculates model prediction lineshape from parameters for the dataset.

    Parameters
    ----------
    paras: dict
        A dictionary containing information about the test
        conditions, properties of the lubricant and material as well
        as model constants used in the interactive friction model.
    i: int
        The suffix of the for the optimisation parameter to have.
    x_input: ndarray
        1D numpy array containing the time or sliding distance data
        for each experimental dataset.
    time_input: bool
        True or False indicating whether the x_set data is time or
        sliding distance data.

    Returns
    -------
    ndarray
        1D numpy array containing the coefficient of friction at
        different times.

    """

    iteration_params = {}

    iteration_params["temperature"] = params[f"temperature_{i+1}"]
    iteration_params["force"] = params[f"force_{i+1}"]
    iteration_params["pressure"] = params[f"pressure_{i+1}"]
    iteration_params["speed"] = params[f"speed_{i+1}"]
    iteration_params["lubricant_thickness"] = params[f"lubricant_thickness_{i+1}"]
    iteration_params["mu0_lubricated"] = params[f"mu_lubricated_0_{i+1}"]
    iteration_params["Q_lubricated"] = params[f"Q_lubricated_{i+1}"]
    iteration_params["mu0_dry"] = params[f"mu_dry_0_1{i+1}"]
    iteration_params["Q_dry"] = params[f"Q_dry_{i+1}"]
    iteration_params["eta_0"] = params[f"eta_0_{i+1}"]
    iteration_params["Q_eta"] = params[f"Q_eta_{i+1}"]

    iteration_params["lambda_1"] = params[f"lambda_1_{i+1}"]
    iteration_params["lambda_2"] = params[f"lambda_2_{i+1}"]
    iteration_params["c"] = params[f"c_{i+1}"]
    iteration_params["k_1"] = params[f"k_1_{i+1}"]
    iteration_params["k_2"] = params[f"k_2_{i+1}"]
    iteration_params["k_3"] = params[f"k_3_{i+1}"]
    # iteration_params["delta_constraint"] = params[f"delta_constraint_{i+1}"]
    # iteration_params["lambda_1_constraint_1"] = params[f"lambda_1_constraint_1_{i+1}"]

    lubricant_thickness = iteration_params['lubricant_thickness'].value

    # For optimisation you need to convert to time input on the x axis.
    return solve_all(x_input, lubricant_thickness, iteration_params, time_input=time_input)


def objective(params, x_set, data, time_input):
    """
    Calculates the total residual for fits of the interactive friction
    model for multiple datasets.

    Parameters
    ----------
    paras: dict
        A dictionary containing information about the test
        conditions, properties of the lubricant and material as well
        as model constants used in the interactive friction model.
    x_set: ndarray
        1D numpy array (Size gives 1D because arrays are of different
        lengths) containing 1D arrays of the different sets of
        experimental time (or possibly sliding distance) data.
    data: ndarray
        1D numpy array (Size gives 1D because arrays are of different
        lengths) containing 1D arrays of the different sets of different
        sets of experimental coefficient of friction data.
    time_input: bool
        True or False indicating whether the x_set data is time or
        sliding distance data.

    Returns
    -------
    ndarray
        1D numpy array containing all residuals.

    """
    # Assume that the number of t sets are the same as number of data sets.

    ndata, *_ = data.shape
    resid = 0.0*data[:]

    for i in range(ndata):
        resid[i] = data[i] - model_dataset(params, i, x_set[i], time_input)

    # Now flatten this to a 1D array, as minimize() requires a 1D array
    # to be optimised. Note that .flatten() doesn't work here because
    # the arrays are of different lengths.
    resid_flattened = np.hstack(resid)

    return resid_flattened


def plot_graphs(plotting_range, base, list_zipped_results, time_input=True, title=None):
    """
    Plots the graphs of the friction model and experimental results
    in one figure with 4 different axes representing changing
    Temperature, Volume, Force and Speed.

    To get the list_zipped_results the following operation on the
    results must be done:
    - list(zip(test_condition_set, sd_measured_set, cof_measured_set))
    THIS MUST BE DONE IN THIS ORDER FOR IT TO WORK AT THE MOMENT.

    Parameters
    ----------
    plotting_range: ndarry
        1D numpy array containing sliding distances/time over which the
        the user wants to plot the results for the interactive friction
        model.

    base: dict
        A dictionary containing the base test conditions.
        Results from this test condition are plotted in all 4 plots.

    list_zipped_results: list
        A list containing a groups of dictionaries with test conditions,
        with a 1D numpy array contining 1D numpy arrays with
        sliding distances and a

    time_input: bool, optional
        By default it assumes that the input is time input.
        If set to True, it knows time input is being used, otherwise
        it will assume sliding distance is used.

    title: str, optional
        Sets the super title of the plot.

    Returns
    -------
    None
        It just plots the graphs.
    """
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))

    # Everything is in the right order, now plot.
    for _, item in enumerate(list_zipped_results):

        if time_input == False:
            cof_computed = solve_all(plotting_range, item[0]['lubricant_thickness'], item[0], time_input=False)
            xlabel = "Sliding Distance/mm"
        elif time_input == True:
            cof_computed = solve_all(plotting_range, item[0]['lubricant_thickness'], item[0], time_input=True)
            xlabel = "Time Elapsed/s"

        # Plot the base in all 4 segments
        if item[0]['temperature'] == base['temperature'] and item[0]['speed'] == base['speed'] and item[0]['force'] == base['force'] and item[0]['lubricant_thickness'] == base['lubricant_thickness']:
            for i in range(2):
                for j in range(2):

                    ax[i,j].xaxis.set_minor_locator(AutoMinorLocator())
                    ax[i,j].yaxis.set_minor_locator(AutoMinorLocator())
                    ax[i,j].set_xlabel(xlabel)
                    ax[i,j].set_ylabel("Coefficient of Friction")
                    if i==0 and j==0:
                        ax[i,j].plot(plotting_range, cof_computed)
                        ax[i,j].scatter(item[1], item[2], label=f"T={item[0]['temperature']}", s=5)
                    elif i==0 and j==1:
                        ax[i,j].plot(plotting_range, cof_computed)
                        ax[i,j].scatter(item[1], item[2], label=f"lubricant_thickness={item[0]['lubricant_thickness']}micrometers", s=5)
                    elif i==1 and j==0:
                        ax[i,j].plot(plotting_range, cof_computed)
                        ax[i,j].scatter(item[1], item[2], label=f"F={item[0]['force']}N", s=5)
                    elif i==1 and j==1:
                        ax[i,j].plot(plotting_range, cof_computed)
                        ax[i,j].scatter(item[1], item[2], label=f"v={item[0]['speed']}mm/s", s=5)


        # Changing T (0,0).
        elif item[0]['speed'] == base['speed'] and item[0]['force'] == base['force'] and item[0]['lubricant_thickness'] == base['lubricant_thickness'] and item[0]['temperature'] != base['temperature']:
            ax[0,0].scatter(item[1], item[2], label=f"T={item[0]['temperature']}", s=5)
            ax[0,0].plot(plotting_range, cof_computed)

        # Changing V (0,1).
        elif item[0]['speed'] == base['speed'] and item[0]['force'] == base['force'] and item[0]['temperature'] == base['temperature'] and item[0]['lubricant_thickness'] != base['lubricant_thickness']:
            ax[0,1].scatter(item[1], item[2], label=f"lubricant_thickness={item[0]['speed']}ss", s=5)
            ax[0,1].plot(plotting_range, cof_computed)

        # Changing F (1,0).
        elif item[0]['speed'] == base['speed'] and item[0]['temperature'] == base['temperature'] and item[0]['lubricant_thickness'] == base['lubricant_thickness'] and item[0]['force'] != base['force']:
            ax[1,0].scatter(item[1], item[2], label=f"F={item[0]['force']}N", s=5)
            ax[1,0].plot(plotting_range, cof_computed)

        # Changing v (1,1).
        elif item[0]['temperature'] == base['temperature'] and item[0]['force'] == base['force'] and item[0]['lubricant_thickness'] == base['lubricant_thickness'] and item[0]['speed'] != base['speed']:
            ax[1,1].scatter(item[1], item[2], label=f"v={item[0]['speed']}mm/s", s=5)
            ax[1,1].plot(plotting_range, cof_computed)

    # Setting the properties for all the axes.
    for i in range(2):
        for j in range(2):
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_xlim(0)
            ax[i,j].legend()

    # Set title, if exists
    if title == None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(title, fontsize=12)

    plt.show()

    return None


def optimise_friction_results(testing_parameters, x_measured_set, cof_measured_set,
                              RANGE_C, RANGE_K1, RANGE_K2, RANGE_K3, RANGE_LAMBDA1, RANGE_LAMBDA2,
                              time_input=True, plot_results=False):
    """
    Takes in arrays of cof, time and a dictionary with testing parameters and
    initial values of the interactive friction model constant parameters.

    Parameters
    ----------
    testing_parameters:


    Returns
    -------
    TODO

    """
    fit_params = Parameters()

    ### Create the sets of fitting parameters, one for each set of data.
    for iy, dictionary in enumerate(testing_parameters, 1):
        # Each item in the list testing_parameters has a dictionary with test conditions.
        fit_params.add(f"temperature_{iy}", value=dictionary["temperature"], vary=False)
        fit_params.add(f"pressure_{iy}", value=dictionary["pressure"], vary=False)
        fit_params.add(f"force_{iy}", value=dictionary["force"], vary=False)
        fit_params.add(f"speed_{iy}", value=dictionary["speed"], vary=False)
        fit_params.add(f"lubricant_thickness_{iy}", value=dictionary["lubricant_thickness"], vary=False)

        fit_params.add(f"mu_lubricated_0_{iy}", value=dictionary["mu0_lubricated"], vary=False)
        fit_params.add(f"Q_lubricated_{iy}", value=dictionary["Q_lubricated"], vary=False)
        fit_params.add(f"mu_dry_0_1{iy}", value=dictionary["mu0_dry"], vary=False)
        fit_params.add(f"Q_dry_{iy}", value=dictionary["Q_dry"], vary=False)
        fit_params.add(f"eta_0_{iy}", value=dictionary["eta_0"], vary=False)
        fit_params.add(f"Q_eta_{iy}", value=dictionary["Q_eta"], vary=False)

        # BACKUP LAMBDA2
        fit_params.add(f"lambda_1_{iy}", value=dictionary["lambda_1"], min=RANGE_LAMBDA1[0], max=RANGE_LAMBDA1[1], vary=True)
        fit_params.add(f"lambda_2_{iy}", value=dictionary["lambda_2"], min=RANGE_LAMBDA2[0], max=RANGE_LAMBDA2[1], vary=True)

        phys_constr = dictionary["blank_roughness"]
        # fit_params.add(f"delta_constraint_{iy}", min=math.log(14.), max=math.log(16.))
        # fit_params.add(f"lambda_2_{iy}", expr=f'delta_constraint_{iy}/(log({phys_constr}*lambda_1_{iy}))')

        lambda_2 = dictionary["lambda_2"]
        # Extra physical constraints
        # fit_params.add(f"lambda_1_constraint_1_{iy}", expr=f'({phys_constr}*lambda_1_{iy})**lambda_2_{iy}', min=0., max=8)
        # fit_params.add(f"lambda_1_constraint_2_{iy}", expr=f'({phys_constr}*lambda_1_{iy})**lambda_2_{iy}', min=16)

        fit_params.add(f"c_{iy}", value=dictionary["c"], min=RANGE_C[0], max=RANGE_C[1])
        fit_params.add(f"k_1_{iy}", value=dictionary["k_1"], min=RANGE_K1[0], max=RANGE_K1[1])
        fit_params.add(f"k_2_{iy}", value=dictionary["k_2"], min=RANGE_K2[0], max=RANGE_K2[1])
        fit_params.add(f"k_3_{iy}", value=dictionary["k_3"], min=RANGE_K3[0], max=RANGE_K3[1])

    # print(phys_constr)
    for iy in range((len(testing_parameters)-1)):
        # To enforce all values to be the same
        fit_params[f"lambda_1_{iy+2}"].expr = "lambda_1_1"
        fit_params[f"lambda_2_{iy+2}"].expr = "lambda_2_1"
        fit_params[f"c_{iy+2}"].expr = "c_1"
        fit_params[f"k_1_{iy+2}"].expr = "k_1_1"
        fit_params[f"k_2_{iy+2}"].expr = "k_2_1"
        fit_params[f"k_3_{iy+2}"].expr = "k_3_1"
        fit_params[f"k_3_{iy+2}"].expr = "k_3_1"
        # fit_params[f"delta_constraint_{iy+2}"].expr = "delta_constraint_1"

    t0 = time.time()
    print("Optimising...")
    # https://groups.google.com/forum/#!topic/lmfit-py/M_t2W3Z6H50 - Customisation for maxiter.
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.basinhopping.html

    # Options: Differential evolution, simulated annealing, basinhopping and more.
    # Necessary to use this method to enable further customisation of optimisation method.

    # https://www2.hawaii.edu/~jonghyun/classes/S18/CEE696/files/14_global_optimization.pdf
    # https://uk.mathworks.com/help/gads/how-simulated-annealing-works.html
    mini = lmfit.Minimizer(objective, fit_params, fcn_args=(x_measured_set, cof_measured_set, time_input), fcn_kws={})
    # result = mini.minimize(method='leastsq', maxfev=100)
    result = mini.minimize(method='basinhopping', niter=25, disp=True)

    t1 = time.time()
    optimisation_time = t1 - t0
    print("Finished Optimising!\n")
    print("Optimisation Results:")
    print(f"    - Optimisation time: {round(optimisation_time, 3)} s")
    print(f"    - Evaluations: {result.nfev}\n")

    # Now print the new values of the optimisation results.
    pretty_table = PrettyTable()
    pretty_table.field_names = ["Parameter", "Old Value", "New Value", "Lower", "Upper"]
    for optimisation_parameter in result.var_names:
        # Remove the _1 from the end of each parameter name.
        parameter_name = optimisation_parameter[:-2]
        old_value = round(fit_params[optimisation_parameter].value, 5)
        new_value = round(result.params[optimisation_parameter].value, 5)
        lower_bound = result.params[optimisation_parameter].min
        upper_bound = result.params[optimisation_parameter].max

        pretty_table.add_row([parameter_name, old_value, new_value,
                              lower_bound, upper_bound])

    print(result.var_names)
    print(pretty_table)

    # Updating parameters is done this way to make it easy to read.
    for optimisation_parameter in result.var_names:
        parameter_name = optimisation_parameter[:-2]
        for dictionary in testing_parameters:
            dictionary[parameter_name] = result.params[optimisation_parameter].value

    if plot_results == True:
        # Plot until the maxmimum value of x given.
        list_zipped_results = (list(zip(testing_parameters, x_measured_set, cof_measured_set)))

        temp = []
        for array in x_measured_set:
            temp.append(np.max(array))

        max_sd_val = max(temp)

        plotting_range = np.linspace(0, max_sd_val, 100)
        base = testing_parameters[0]
        plot_graphs(plotting_range, base, list_zipped_results, time_input=time_input)

    print()
    return testing_parameters
    # Ctrl+Shift+L to select all of the same instance.


def extract_data_optimisation(group_id_chosen, avg_load_loc=None):
    """
    Calculates model prediction lineshape from parameters for the dataset.

    Parameters
    ----------
    group_id_chosen: int
        The value of the groups chosen for optimisation (e.g. 1, 2, 3).
        This is based on grouping from tests.
    avg_load_loc: str, optional
        The relative/absolute location of the folder containing the files with
        the average values from tests. If not specified, it is assumed that
        the average files are all in the same location as the script.

    Returns
    -------
    ndarray
        1D numpy array containing the coefficient of friction at
        different times.

    (testing_parameters_all, time_measured_set, sd_measured_set, cof_measured_set)

    """
    print("")
    print("Extracting data from SQL table...")
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
              .where(experiments.c.select == True)\
              .where(experiments.c.select == True)

    ResultProxy  = con.execute(query)
    results = ResultProxy.fetchall()

    if len(results) != 0:
        df = pd.DataFrame(results)
        columns = results[0].keys()
        df.columns = columns

        # Remove duplicates of the same temperature, speed, force and volume
        df = df.drop_duplicates(subset=["temperature", "force", "speed", "volume"])

        if df.any().all() == False:
            # If there are any empty values (False means empty)
            print("Empty Columns:")
            for col in columns:
                # Go through all the columns to determine which is the empty ones.
                if df.any()[col] == False:
                    print(f"    - {col}")

            print(f"Aborting test, fill in the null columns first for lubricant_id {df.lubricant_id.values}.")
            print("")
            return 0

        else:
            # Extract the testing_parameters, put into a dictionary,
            # and add it to this list.
            testing_parameters_all = []

            # Read the csv file and append the required information to
            # these lists.
            cof_measured_set = []
            time_measured_set = []
            sd_measured_set = []

            # Now transform into the required form for optimisation.
            for _, row in df.iterrows():
                # Create a dictionary from test condition
                temp_dictionary = {}
                temp_dictionary['temperature'] = float(row['temperature'])
                temp_dictionary['force'] = float(row['force'])
                temp_dictionary['P'] = float(row['pressure'])
                temp_dictionary['speed'] = float(row['speed'])
                temp_dictionary['speed'] = float(row['volume'])
                temp_dictionary['lubricant_thickness'] = float(row['film_thickness'])
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
                temp_dictionary['lubricant_id'] = int(row['lubricant_id'])

                testing_parameters_all.append(temp_dictionary)

                # We just need the time, sliding distance, coefficient of friction columns
                f = row['avg_file_name']

                # Now load the data as required.
                if avg_load_loc == None:
                    # Assume all files are in the current directory.
                    df_averaged_data = pd.read_csv(f)
                else:
                    # Use the location specified.
                    df_averaged_data = pd.read_csv(f"{avg_load_loc}\\{f}")

                # print(df_averaged_data)
                # print(df_averaged_data['time_elapsed_(s)'].values.shape)

                cof_measured_set.append(df_averaged_data['coefficient_of_friction'].values)
                time_measured_set.append(df_averaged_data['time_elapsed_(s)'].values)
                sd_measured_set.append(df_averaged_data['sliding_distance_(mm)'].values)

            # Now convert the cof, time and sd lists to numpy ndarrays
            cof_measured_set = np.array(cof_measured_set)
            time_measured_set = np.array(time_measured_set)
            sd_measured_set = np.array(sd_measured_set)

            con.close()
            return (testing_parameters_all, time_measured_set, sd_measured_set, cof_measured_set)
    else:
        print("No complete results to optimise with")
        con.close()
        return 0


def update_model_parameters(group_id, params):
    """ Using the name of the lubricant, the correct row will be updated. """

    engine = create_engine('sqlite:///friction_model.db')
    con = engine.connect()
    metadata = sqlalchemy.MetaData()

    # Load the tables
    optimisation_groups = sqlalchemy.Table('optimisation_groups', metadata, autoload=True, autoload_with=engine)

    print("Group id", group_id)
    # Update database with averaged file name
    u = update(optimisation_groups).where(optimisation_groups.c.group_id == int(group_id))
    u = u.values(lambda_1=params['lambda_1'], lambda_2=params['lambda_2'], c=params['c'],
                 k_1=params['k_1'], k_2=params['k_2'], k_3=params['k_3'])

    con.execute(u)
    con.close()


def optimisation_friction_model(group_id_chosen=1):
    """
    Run this function to optimise.
    """
    ### Extracting averaged for optimisation
    # Choose the optimisation group & file loading location
    avg_load_loc = "friction_averaged_results"

    results = extract_data_optimisation(group_id_chosen, avg_load_loc)

    if results == 0:
        print("Optimisation failed, ending script.")
        return 0
    else:
        # Unpack the variables.
        print("Unpacking the variables...")
        testing_parameters_all, time_measured_set, sd_measured_set, cof_measured_set = results
        num_averaged_results = len(testing_parameters_all)
        print(f"There are {num_averaged_results} datasets to optimise")

    ### SAMPLE TESTING DATA
    # cof_measured_set, time_measured_set, sd_measured_set, testing_parameters_all = load_sample_data_opt()

    # You need to convert to time and coefficient of friction data before processing.
    # You only need the first item in the list as the parameters will be the same for al of them.
    paras_solved1 = optimise_friction_results(testing_parameters_all, sd_measured_set, cof_measured_set, plot_results=True, time_input=False)
    lubricant_id = paras_solved1[0]['lubricant_id']

    while True:
        print()
        print("Do you want to overwrite the old parameter values? (Y/N)")
        print(f"Lubricant ID: {lubricant_id}")

        break


def plot_results_modified(group_id_chosen=1):
    """ To test that plotting works """
    ### Extracting averaged for optimisation
    # Choose the optimisation group & file loading location
    avg_load_loc = "friction_averaged_results"

    results = extract_data_optimisation(group_id_chosen, avg_load_loc)

    if results == 0:
        print("Manual fitting failed, ending script.")
        return 0
    else:
        # Unpack the variables.
        print("Unpacking the variables...")
        testing_parameters_all, _, sd_measured_set, cof_measured_set = results
        num_averaged_results = len(testing_parameters_all)
        print(f"There are {num_averaged_results} datasets")

    # Assumes that the first test done is the base (Otherwise the user needs to manually pick)
    base = testing_parameters_all[0]
    ### User defined parameters
    ### k_2 is now < 1 to indicate low speed region. Original values: lambda_1 = 20, lambda_2 = 1.1, k1 = 2.05, k2 = 2.98, k3 = 5.3, c = 0.012
    # constant_fitting_params = dict(mu0_lubricated = 1.6907363829537443, Q_lubricated = 9141.506836756891, mu0_dry = 10.942250929819629, Q_dry = 9368.85126706061, eta_0 = 0.12, Q_eta = 11930, blank_roughness = 0.3, pin_roughness = 0.8)
    plotting_range = np.linspace(0, 88, 420)

    # NOTE: Increasing k1 increases the distance for force the most and makes the sd transition later (As it is P**k1)
    variable_fitting_params = dict(lambda_1 = 20, lambda_2 = 1.8, c = 15, k_1 = 1.6, k_2 = 0.7, k_3 = 3.2)

    testing_parameters_all = [{**sql_fitting_params.copy(), **variable_fitting_params} for sql_fitting_params in testing_parameters_all]

    # Plotting using the advanced function
    list_zipped_results = list(zip(testing_parameters_all, sd_measured_set, cof_measured_set))
    plot_graphs(plotting_range=plotting_range, base=base, list_zipped_results=list_zipped_results, time_input=False, title="Interactive Friction Model")
    return 1


def test_plotting():
    """ To test that the plotting function works as required. Call the function to see results. """
    # To test that the plotting works
    current_test_parameters_1 = dict(temperature=450, force=5, pressure=0.34, speed=50, lubricant_thickness=25, lubricant_volume=23.2,
                                     mu0_lubricated = 1.69073, Q_lubricated = 9141.50683,
                                     mu0_dry = 10.94225, Q_dry = 9368.85126, eta_0 = 0.12,
                                     Q_eta = 11930, lambda_1 = 40, lambda_2 = 1.5, c = 0.00847,
                                     k_1 = 1.52, k_2 = 2.77, k_3 = 4.8)

    plotting_range = np.linspace(0, 100, 420)
    ans = solve_all(plotting_range, current_test_parameters_1['lubricant_thickness'], current_test_parameters_1, time_input=False)
    plt.plot(plotting_range, ans)
    plt.xlabel("Sliding Distance / mm")
    plt.ylabel("Coefficient of Friction")
    plt.xlim(0)
    plt.ylim(0)
    plt.grid()
    plt.show()


def manual_fitting_slider(testing_parameters_all, sd_measured_set, cof_measured_set):
    """ Manual Fitting with a Slider (1 Plot) """

    ### UP TO THE USER TO DECIDE THE NUMBER OF POINTS
    num_points = 100

    ### THE USER MAY CHANGE THE MAXIMUM PLOTTING DISTANCE
    max_sd_val = 90

    ### THE USER MAY CHANGE THE RANGE FOR THE SLIDERS
    LAMBDA1_MINMAX = [0.1, 100.0]
    LAMBDA2_MINMAX = [0.1, 8.0]
    k1_MINMAX = [0.1, 30.0]
    k2_MINMAX = [0.1, 3.0]
    k3_MINMAX = [1.0, 8.0]
    c_MINMAX = [0.001, 0.75]

    plotting_range = np.linspace(0, max_sd_val, num_points)

    y_vals_list = []
    for test_parameter in testing_parameters_all:
        y_vals = solve_all(plotting_range, test_parameter['lubricant_thickness'], test_parameter, time_input=False)
        y_vals_list.append(y_vals)

    plot_list = []

    lubricant_id = testing_parameters_all[0]['lubricant_id']
    group_id = testing_parameters_all[0]['group_id']
    blank_roughness = testing_parameters_all[0]['blank_roughness']


    fig, ax = plt.subplots(figsize=(11, 7.37))

    ax.set_title("Interactive Friction Model", fontsize=12)
    ax.set_xlim(0, max_sd_val)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Coefficient of Friction")
    ax.set_xlabel("Sliding Distance / mm")
    fig.subplots_adjust(top=0.95, bottom=0.45, right=0.65, left=0.08)


    for idx, y_vals in enumerate(y_vals_list):
        T = testing_parameters_all[idx]['temperature']
        v = testing_parameters_all[idx]['speed']
        V = testing_parameters_all[idx]['speed']
        F = testing_parameters_all[idx]['force']

        # l, = plt.plot(plotting_range, y_vals, lw=2, label=f'T={T}')
        l, = plt.plot(plotting_range, y_vals, lw=2, label=f'T={T}, v={v}, V={V}, F={F}')
        ax.scatter(sd_measured_set[idx], cof_measured_set[idx], s=8)
        plot_list.append(l)

    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=True, ncol=1)

    ### Store in the storage object
    plot_storage = PlottingStorage()
    plot_storage.store_plotting_range(plotting_range)
    plot_storage.store_testing_parameters_all(testing_parameters_all)
    plot_storage.store_y_vals_list(y_vals_list)
    plot_storage.store_plot_list(plot_list)
    plot_storage.store_blank_roughness(blank_roughness)
    plot_storage.store_lubricant_id(lubricant_id)
    plot_storage.store_group_id(group_id)

    # Make checkbuttons with all plotted lines with correct visibility
    rax = plt.axes([0.67, 0.1, 0.30, 0.28])
    labels = [str(line.get_label()) for line in plot_storage.plot_list]
    visibility = [line.get_visible() for line in plot_storage.plot_list]
    check = CheckButtons(rax, labels, visibility)

    def toggle_visibility(label):
        index = labels.index(label)
        plot_storage.plot_list[index].set_visible(not plot_storage.plot_list[index].get_visible())
        plt.draw()

    check.on_clicked(toggle_visibility)

    # Define an axes area and draw a slider in it (x6)
    lambda1_slider_ax  = fig.add_axes([0.1, 0.35, 0.5, 0.03])
    lambda1_slider = Slider(lambda1_slider_ax, 'lambda1', LAMBDA1_MINMAX[0], LAMBDA1_MINMAX[1], valinit=testing_parameters_all[0]['lambda_1'])
    lambda2_slider_ax = fig.add_axes([0.1, 0.30, 0.5, 0.03])
    lambda2_slider = Slider(lambda2_slider_ax, 'lambda2', LAMBDA2_MINMAX[0], LAMBDA2_MINMAX[1], valinit=testing_parameters_all[0]['lambda_2'])
    k1_slider_ax = fig.add_axes([0.1, 0.25, 0.5, 0.03])
    k1_slider = Slider(k1_slider_ax, 'k1 (P)', k1_MINMAX[0], k1_MINMAX[1], valinit=testing_parameters_all[0]['k_1'])
    k2_slider_ax = fig.add_axes([0.1, 0.20, 0.5, 0.03])
    k2_slider = Slider(k2_slider_ax, 'k2 (v)', k2_MINMAX[0], k2_MINMAX[1], valinit=testing_parameters_all[0]['k_2'])
    k3_slider_ax = fig.add_axes([0.1, 0.15, 0.5, 0.03])
    k3_slider = Slider(k3_slider_ax, 'k3 (T)', k3_MINMAX[0], k3_MINMAX[1], valinit=testing_parameters_all[0]['k_3'])
    c_slider_ax = fig.add_axes([0.1, 0.1, 0.5, 0.03])
    c_slider = Slider(c_slider_ax, 'c (scaling)', c_MINMAX[0], c_MINMAX[1], valinit=testing_parameters_all[0]['c'])


    def reset_button_on_clicked(mouse_event):
        lambda1_slider.reset()
        lambda2_slider.reset()
        k1_slider.reset()
        k2_slider.reset()
        k3_slider.reset()
        c_slider.reset()

    def save_button_on_clicked(mouse_event):
        lambda_1 = lambda1_slider.val
        lambda_2 = lambda2_slider.val
        k_1 = k1_slider.val
        k_2 = k2_slider.val
        k_3 = k3_slider.val
        c = c_slider.val

        # Update SQL Table with new values
        group_id = plot_storage.group_id
        dictionary_parameters = dict(lambda_1=lambda_1, lambda_2=lambda_2, k_1=k_1, k_2=k_2, k_3=k_3, c=c)
        update_model_parameters(group_id, dictionary_parameters)

    def update_graph_lambdas(val):
        """ Update the graph LAMBDA """
        lambda_1 = lambda1_slider.val
        lambda_2 = lambda2_slider.val
        k_1 = k1_slider.val
        k_2 = k2_slider.val
        k_3 = k3_slider.val
        c = c_slider.val

        blank_roughness = plot_storage.blank_roughness

        constraint1 = (blank_roughness*lambda_1)**lambda_2 < 8
        constraint2 = (blank_roughness*lambda_1)**lambda_2 > 16

        if constraint1 or constraint2:
            print("Physical Constraint Failed (Change Lambda1 and Lambda2)")
        else:
            print("Physical Constraint Passed")

        updated_params = dict(lambda_1 = lambda_1, lambda_2 = lambda_2, k_1 = k_1, k_2 = k_2, k_3 = k_3, c = c)

        # Store the results in an object to be used in the functions (A Hack to make it work).
        testing_parameters_all = plot_storage.testing_parameters_all
        y_vals_list = plot_storage.y_vals_list
        plot_list = plot_storage.plot_list
        plotting_range = plot_storage.plotting_range

        # Update the graph
        for idx, current_test_parameters in enumerate(testing_parameters_all):
            current_test_parameters = {**current_test_parameters, **updated_params}
            ans = solve_all(plotting_range, current_test_parameters['lubricant_thickness'], current_test_parameters, time_input=False)
            plot_list[idx].set_ydata(ans)

        # redraw canvas while idle
        fig.canvas.draw_idle()


    def update_graph(val):
        """ Update the graph """
        lambda_1 = lambda1_slider.val
        lambda_2 = lambda2_slider.val
        k_1 = k1_slider.val
        k_2 = k2_slider.val
        k_3 = k3_slider.val
        c = c_slider.val

        updated_params = dict(lambda_1 = lambda_1, lambda_2 = lambda_2, k_1 = k_1, k_2 = k_2, k_3 = k_3, c = c)

        # Store the results in an object to be used in the functions (A Hack to make it work).
        testing_parameters_all = plot_storage.testing_parameters_all
        y_vals_list = plot_storage.y_vals_list
        plot_list = plot_storage.plot_list
        plotting_range = plot_storage.plotting_range

        # Update the graph
        for idx, current_test_parameters in enumerate(testing_parameters_all):
            current_test_parameters = {**current_test_parameters, **updated_params}
            ans = solve_all(plotting_range, current_test_parameters['lubricant_thickness'], current_test_parameters, time_input=False)
            plot_list[idx].set_ydata(ans)

        # redraw canvas while idle
        fig.canvas.draw_idle()

    # call update function on slider value change
    lambda1_slider.on_changed(update_graph_lambdas)
    lambda2_slider.on_changed(update_graph_lambdas)
    k1_slider.on_changed(update_graph)
    k2_slider.on_changed(update_graph)
    k3_slider.on_changed(update_graph)
    c_slider.on_changed(update_graph)

    # Add a button for resetting the parameters
    reset_button_ax = fig.add_axes([0.42, 0.0271, 0.1, 0.04])
    reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')

    save_params_ax = fig.add_axes([0.15, 0.0271, 0.2, 0.04])
    save_button = Button(save_params_ax, 'Save Parameters', hovercolor='0.975')

    reset_button.on_clicked(reset_button_on_clicked)
    save_button.on_clicked(save_button_on_clicked)

    plt.show()


class PlottingStorage():
    """ A storage object for plotting modules """

    def __init__(self):
        self.description = "Stores results for plotting"

    def store_plotting_range(self, plotting_range):
        self.plotting_range = plotting_range

    def store_testing_parameters_all(self, testing_parameters_all):
        self.testing_parameters_all = testing_parameters_all

    def store_y_vals_list(self, y_vals_list):
        self.y_vals_list = y_vals_list

    def store_plot_list(self, plot_list):
        self.plot_list = plot_list

    def store_blank_roughness(self, blank_roughness):
        self.blank_roughness = blank_roughness

    def store_lubricant_id(self, lubricant_id):
        self.lubricant_id = lubricant_id

    def store_group_id(self, group_id):
        self.group_id = group_id

if __name__ == "__main__":
    # https://github.com/matplotlib/matplotlib/issues/7991/ - Look into this for plotting.
    # https://riptutorial.com/matplotlib/example/23577/interactive-controls-with-matplotlib-widgets - Plots with widgets.
    # https://stackoverflow.com/questions/6697259/interactive-matplotlib-plot-with-two-sliders - Plots with slider, button and radio button widgets.
    # https://matplotlib.org/3.1.1/gallery/widgets/textbox.html - Plots with input widget.
    # https://stackoverflow.com/questions/25954522/accessing-an-object-outside-of-a-nested-function-in-python - A method to streamline code.

    print("Starting the script")

    # FEA_data_file = "lub_evaluation_data.mat"
    # load_FEA_data(FEA_data_file)

    # test_plotting()
    # lubricant_evaluation()
