import numpy as np
import time
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import int32, float32, double, typed, typeof, boolean
import numba

# # https://stackoverflow.com/questions/38682260/how-to-nest-numba-jitclass
# # https://stackoverflow.com/questions/55522355/is-there-a-way-to-include-string-attributes-in-numba-jitclass
# https://stackoverflow.com/questions/47667906/how-to-specify-numba-jitclass-when-the-classs-attribute-contains-another-class
# conditions_type = deferred_type()
# conditions_type.define(StaticExperimentalConditions.class_type.instance_type)

class Experiment:
    def __init__(self, static_conditions, variable_conditions) -> None:

        self.static_conditions = static_conditions
        self.variable_conditions = variable_conditions

        self.condition_id = None
        self.group_id = None
        self.duplicate_number = None
        self.filename = None

spec_variable_experimental_conditions = [
    ('temperature', float32),
    ('force', float32),
    ('pressure', float32),
    ('speed', float32),
    ('lubricant_thickness', float32),
    ('equiv_solid_thickness', float32)
]

@jitclass(spec_variable_experimental_conditions)
class VariableExperimentalConditions:
    def __init__(self, temperature, force, pressure, speed, lubricant_thickness=0, equiv_solid_thickness=0):
        self.temperature = temperature
        self.force = force
        self.pressure = pressure
        self.speed = speed
        self.lubricant_thickness = lubricant_thickness
        self.equiv_solid_thickness = equiv_solid_thickness

spec_lubricant = [
    ('id', int32),
    ('name', numba.types.string),
    ('includes_liquid', boolean),
    ('includes_solid', boolean),
    ('density_liquid', float32),
    ('avg_hardness_solid', float32),
]

@jitclass(spec_lubricant)
class Lubricant():

    def __init__(self, id, name, includes_liquid, includes_solid, density_liquid=0, avg_hardness_solid=0):
        self.id = id
        self.name = name
        self.includes_liquid = includes_liquid
        self.includes_solid = includes_solid
        self.density_liquid = density_liquid
        self.avg_hardness_solid = avg_hardness_solid

        if self.includes_liquid:
            # check that it has liquid density
            if density_liquid == 0:
                raise Exception('If includes liquid, density_liquid required!')

            # Assumes constant density (Even though it changes with temperature)
            # Density Units: mg/mm^3
            self.density_liquid = density_liquid

        if self.includes_solid:
            # check that is has avg_hardness_solid
            if avg_hardness_solid == 0:
                raise Exception('If includes solid, avg_hardness_solid required!')

            # Average Hardness Units: GPa
            self.avg_hardness_solid = avg_hardness_solid

spec_static_experimental_conditions = [
    ('lubricant', Lubricant.class_type.instance_type),
    ('pin_material', numba.types.string),
    ('pin_roughness', float32),
    ('blank_material', numba.types.string),
    ('blank_roughness', float32),
    ('blank_thickness', float32),
    ('coating_material', numba.types.string),
    ('coating_thickness', float32),
    ('coating_roughness', float32)
]

@jitclass(spec_static_experimental_conditions)
class StaticExperimentalConditions:

    def __init__(self, lubricant, pin_material, pin_roughness, blank_material,
                 blank_roughness, blank_thickness, coating_material,
                 coating_thickness, coating_roughness) -> None:

        self.lubricant = lubricant

        self.pin_material = pin_material
        self.pin_roughness = pin_roughness

        self.blank_material = blank_material
        self.blank_roughness = blank_roughness
        self.blank_thickness = blank_thickness

        self.coating_material = coating_material
        self.coating_thickness = coating_thickness
        self.coating_roughness = coating_roughness

spec_friction_liquid_variable_conditions_data = [
    ('temperature', int32),
    ('force', int32),
    ('pressure', float32),
    ('speed', float32),
    ('lubricant_thickness', float32),
    ('x_data', double[:]),
    ('y_data', double[:]),
]

# @jitclass(spec_friction_liquid_variable_conditions_data)
class FrictionLiquidVariableConditionsData():

    def __init__(self, temperature, force, pressure, speed, lubricant_thickness, x_data, y_data) -> None:
        # super().__init__(x_data, y_data)
        # In this case x_data will all be the same length. The same with y_data
        self.temperature = temperature
        self.force = force
        self.pressure = pressure
        self.speed = speed
        self.lubricant_thickness = lubricant_thickness

        self.x_data = x_data
        self.y_data = y_data

spec_friction_liquid_solid_variable_conditions_data = [
    ('temperature', int32),                # A simple scalar value (float)
    ('force', int32),                      # A simple scalar value (float)
    ('pressure', float32),                 # A simple scalar value (float)
    ('speed', float32),                    # A simple scalar value (float)
    ('lubricant_thickness', float32),      # A simple scalar value (float)
    ('equiv_solid_thickness', float32),    # A simple scalar value (float)
    ('x_data', double[:]),                 # An array of doubles (values can get very big)
    ('y_data', double[:]),                 # An array of doubles (values can get very big)
]

# @jitclass(spec_friction_liquid_solid_variable_conditions_data)
class FrictionLiquidSolidVariableConditionsData():

    def __init__(self, temperature, force, pressure, speed, lubricant_thickness, equiv_solid_thickness, x_data, y_data) -> None:
        # super().__init__(x_data, y_data)
        # In this case x_data will all be the same length. The same with y_data
        self.temperature = temperature
        self.force = force
        self.pressure = pressure
        self.speed = speed
        self.lubricant_thickness = lubricant_thickness
        self.equiv_solid_thickness = equiv_solid_thickness

        self.x_data = x_data
        self.y_data = y_data

spec_friction_model_parameters_liquid = [
    ('mu0_lubricated', float32),
    ('Q_lubricated', float32),
    ('mu0_dry', float32),
    ('Q_dry', float32),
    ('eta_0', float32),
    ('Q_eta', float32),
    ('lambda_1', float32),
    ('lambda_2', float32),
    ('c', float32),
    ('k_1', float32),
    ('k_2', float32),
    ('k_3', float32),
]

@jitclass(spec_friction_model_parameters_liquid)
class FrictionModelParametersLiquid():

    def __init__(self, mu0_lubricated, Q_lubricated, mu0_dry, Q_dry, eta_0, Q_eta,
                 lambda_1, lambda_2, c, k_1, k_2, k_3) -> None:

        self.mu0_lubricated = mu0_lubricated
        self.Q_lubricated = Q_lubricated
        self.mu0_dry = mu0_dry
        self.Q_dry = Q_dry
        self.eta_0 = eta_0
        self.Q_eta = Q_eta

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.c = c
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3

spec_friction_model_parameters_liquid_solid = [
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

@jitclass(spec_friction_model_parameters_liquid_solid)
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


class FrictionModelLiquid():
    """
    The FrictionModel1 object
    """
    R = 8.314

    def __init__(self, conditions, parameters) -> None:
        self.static_conditions = conditions
        self.parameters = parameters
        self.model_name = 'Friction model liquid'
        self.time_input = False

    def plot_cof_prediction(self, prediction_range, temperature, force, pressure, speed, lubricant_thickness) -> None:
        """ Plots the predicted coefficient of friction given a range of x values

        This function will assume time input is given. Change the time_input
        flag to True if the input is time.

        Args:
            prediction_range (np.ndarray): Prediction range to for cof to be
                plotted for. If time, in seconds. If sliding distance, in mm.
        """

        cof_predicted = self.predict(prediction_range, temperature, force, pressure, speed, lubricant_thickness)
        self._plot_result('coefficient_of_friction', prediction_range, cof_predicted, temperature, force, pressure, speed, lubricant_thickness)

    def plot_lubricant_thickness_prediction(self, prediction_range, temperature, force, pressure, speed, lubricant_thickness):
        if self.time_input==False:
            prediction_range = prediction_range/speed

        eta = self._predict_eta(temperature)
        lubricant_thickness_predicted = self.predict_lubricant_thickness(lubricant_thickness, prediction_range, force, pressure, speed, eta)
        self._plot_result('lubricant_thickness', prediction_range, lubricant_thickness_predicted, temperature, force, pressure, speed, lubricant_thickness)

    def predict(self, prediction_range, temperature, force, pressure, speed, lubricant_thickness) -> np.ndarray:
        if self.time_input==False:
            prediction_range = prediction_range/speed

        cof_lubricated = self._predict_cof_lubricated(temperature)
        cof_dry = self._predict_cof_dry(temperature)
        eta = self._predict_eta(temperature)

        h_array = self.predict_lubricant_thickness(lubricant_thickness, prediction_range, force, pressure, speed, eta)
        h_array = np.nan_to_num(h_array, nan=0.0)
        beta_array = np.exp(-(self.parameters.lambda_1 * h_array) ** self.parameters.lambda_2)
        beta_array = np.nan_to_num(beta_array, nan=1.0)
        cof_predicted = (1 - beta_array) * cof_lubricated + beta_array * cof_dry

        return cof_predicted

    def predict_lubricant_thickness(self, lubricant_thickness, prediction_range, force, pressure, speed, eta):
        h_array = odeint(self._lubricant_thickness_rate,
                         lubricant_thickness,
                         prediction_range, args=(force, pressure, speed, eta))

        return h_array

    def _plot_result(self, prediction_type, prediction_range, y_predicted, temperature, force, pressure, speed, lubricant_thickness) -> None:

        if prediction_type == 'coefficient_of_friction':
            ylabel = 'Coefficient of Friction'
        elif prediction_type == 'lubricant_thickness':
            ylabel = 'Lubricant Thickness / micrometers'
        else:
            ylabel = None

        _, ax = plt.subplots()
        ax.plot(prediction_range, y_predicted)

        ax.set_title('Friction Model Liquid Prediction')
        if self.time_input==True:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel('Sliding Distance (mm)')
            ax.set_ylabel(ylabel)

        textstr = '\n'.join((f'Temperature: {temperature} (°C)', f'Force: {force} (N)',
            f'Pressure: {pressure} (MPa)', f'Speed: {speed} (mm/s)',
            f'Lubricant Thickness: {lubricant_thickness} (micrometers)'))
        props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)

        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top', bbox=props)
        ax.set_xlim(0)
        ax.set_ylim(0)
        plt.show()

    def _lubricant_thickness_rate(self, lubricant_thickness, t, _, pressure, speed, eta) -> float:
        h_rate = - lubricant_thickness * \
            self.parameters.c * (pressure ** self.parameters.k_1) * \
            (speed ** self.parameters.k_2) / \
            (eta ** self.parameters.k_3)

        return h_rate

    def _calculate_beta(self, lubricant_thickness) -> float:
        beta = np.exp(-(self.parameters.lambda_1 * lubricant_thickness) \
            ** self.parameters.lambda_2)

        return beta

    def _predict_cof_lubricated(self, temperature) -> float:
        """ Predicts the coefficient of friction (full film lubrication)

        Returns:
            float: Full film lubrication coefficient of friction
        """
        cof_lubricated = self.parameters.mu0_lubricated * \
            np.exp(-self.parameters.Q_lubricated/(self.R * self._convert_to_kelvin(temperature)))

        return cof_lubricated

    def _predict_cof_dry(self, temperature) -> float:
        """ Predicts the coefficient of friction (dry sliding)

        Returns:
            float: Dry sliding coefficient of friction
        """
        cof_dry = self.parameters.mu0_dry * \
            np.exp(-self.parameters.Q_dry/(self.R * self._convert_to_kelvin(temperature)))

        return cof_dry

    def _predict_eta(self, temperature) -> float:
        """ Predicts the coefficient of friction

        Args:
            temperature (Union[int, float]): test temperature

        Returns:
            float: Converted temperature in Kelvins
        """
        eta = self.parameters.eta_0 * \
            np.exp(self.parameters.Q_eta/(self.R * self._convert_to_kelvin(temperature)))

        return eta

    def _convert_to_kelvin(self, temperature) -> float:
        """ Converts temperature from degrees to Kelvin

        Args:
            temperature (Union[int, float]): test temperature

        Returns:
            float: Converted temperature in Kelvins
        """
        return temperature + 273.1

spec = [
    ('static_conditions', StaticExperimentalConditions.class_type.instance_type),
    ('parameters', FrictionModelParametersLiquidSolid.class_type.instance_type),
    ('model_name', numba.types.string),
    ('time_input', boolean),
    ('dt', float32),
    ('R', float32),
    ('coating_roughness', float32)
]

@jitclass(spec)
class FrictionModelLiquidSolid():
    """
    The FrictionModel2 object
    """

    def __init__(self, static_conditions, parameters) -> None:
        self.static_conditions = static_conditions
        self.parameters = parameters
        self.model_name = 'friction_model_liquid_solid'
        self.time_input = False
        self.dt = 0.0005
        self.R = 8.314

    def predict(self, prediction_range, temperature, force, pressure, speed, lubricant_thickness, equiv_solid_thickness) -> np.ndarray:
        if self.time_input==False:
            prediction_range = prediction_range/speed

        # We want a set dt array for this prediction range: 0.0005 seconds, 5000 pts -> 2.5 seconds
        max_time = np.max(prediction_range)
        t_array = np.arange(0, max_time, self.dt)

        temperature = self._convert_to_kelvin(temperature)
        cof_low = self._predict_cof_low(temperature)
        cof_dry = self._predict_cof_dry(temperature)
        contact_area = self._calculate_approx_contact_area(force, pressure)
        K_wear = self.parameters.K * np.exp(-self.parameters.Q_K/(self.R * temperature))

        # Initialise arrays
        alpha_t = np.zeros_like(t_array)
        alpha_t_dot = np.zeros_like(t_array)
        h2_dot = np.zeros_like(t_array)
        h_liquid = np.zeros(t_array.shape[0] + 1)
        H_s = np.zeros_like(t_array)
        h_solid_dot = np.zeros_like(t_array)
        h_solid = np.zeros(t_array.shape[0] + 1)

        # Set initial values
        h_liquid[0] = lubricant_thickness
        h_solid[0] = equiv_solid_thickness

        for i in range(t_array.shape[0]):

            if h_liquid[i] > 0:
                # Instantaneous weight of PEG
                alpha_t[i] = h_liquid[i] * contact_area * self.static_conditions.lubricant.density_liquid

                # Decrease of h_liquid due to degradation
                alpha_t_dot[i] = self.parameters.D * \
                    np.exp(-self.parameters.Q_alpha/(self.R*temperature)) * (alpha_t[i]) \
                    ** self.parameters.k_alpha
                alpha_step = alpha_t_dot[i] * self.dt
                h1_step = -alpha_step / (contact_area * self.static_conditions.lubricant.density_liquid)

                # Decrease of h_liquid due to smear on the wear track
                h2_dot[i] = -h_liquid[i] * (self.parameters.c*(pressure ** self.parameters.k_1) \
                    * (speed **self.parameters.k_2))
                h2_step = h2_dot[i] * self.dt

                # Instantaneous h_liquid
                h_liquid[i+1] = h_liquid[i] + h1_step + h2_step
            else:
                h_liquid[i+1] = 0

            # Calculation of graphite coating thickness
            if h_solid[i] > 0:
                H_s[i] = self.static_conditions.lubricant.avg_hardness_solid/(h_solid[i]**self.parameters.k_4)
                h_solid_dot[i] = K_wear * (pressure**self.parameters.k_5* \
                    (speed**self.parameters.k_3)/(H_s[i]))
                h_solid[i+1] = h_solid[i] - h_solid_dot[i]*self.dt
            else:
                h_solid[i+1] = 0

        # Calculation of coefficient_of_friction
        h_grease = h_solid + h_liquid
        h_grease[np.isnan(h_grease)] = 0.0
        beta = np.exp(-(self.parameters.lambda_1*(h_grease)*1000) ** self.parameters.lambda_2)
        beta[np.isnan(beta)] = 1.0
        cof_predicted = (1-beta) * cof_low + beta * cof_dry

        # Alternative to using isnan, is to use:
        # mini = lmfit.Minimizer(residuals, p, nan_policy='omit')
        # But that would mean some datapoints (particularly later ones) are omitted

        # Get closest values (TODO if necessary: A more efficient method can be implemented)
        # closest_idxs = [self.find_nearest_idx(t_array, value) for value in prediction_range]
        closest_idxs = []
        for value in prediction_range:
            closest_idxs.append(self.find_nearest_idx(t_array, value))

        cof_chosen = cof_predicted[np.array(closest_idxs)]

        return cof_chosen

    def _convert_to_kelvin(self, temperature) -> float:
        """ Converts temperature from degrees to Kelvin

        Args:
            temperature (Union[int, float]): test temperature

        Returns:
            float: Converted temperature in Kelvins
        """
        return temperature + 273.1

    def _calculate_approx_contact_area(self, force, pressure):
        wear_track_width = (8*force/(np.pi*pressure*1000))**0.5
        contact_area = (np.pi*wear_track_width**2)/4

        return contact_area

    def find_nearest_idx(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _predict_cof_low(self, temperature) -> float:

        if temperature == self._convert_to_kelvin(450):
            return 0.18
        elif temperature == self._convert_to_kelvin(400):
            return 0.14
        elif temperature == self._convert_to_kelvin(300):
            return 0.12
        else:
            # NOTE: These temperatures are specific to these test conditions
            # If another static test condition is used, different values should be used.
            raise Exception("Temperature used not one that is supported! Pick from 300, 400 and 450.")

    def _predict_cof_dry(self, temperature) -> float:
        if temperature == self._convert_to_kelvin(450):
            return 1.5
        elif temperature == self._convert_to_kelvin(400):
            return 1.5
        elif temperature == self._convert_to_kelvin(300):
            return 1.5
        else:
            # NOTE: These temperatures are specific to these test conditions
            # If another static test condition is used, different values should be used.
            raise Exception("Temperature used not one that is supported! Pick from 300, 400 and 450.")

def predict_model_old():

    lambda1 = 1.69633723376872; lambda2 = 0.802696062360457; c = 0.0130799271358491; k1 = 0.801289664383808
    k2 = 2.07037498052891; D = 2.66926336189041*10000; Qa = 4.98493351507642*10000; ka = 1.00674951577508; k3 = 2.81418025418379
    k4 = 1.39617733322863; K = 1.07360693295872*10000; Qk = 7.40524741745115*10000
    h_l = 0.025; h_s = 0.01
    k5 = 1.85796986280561

    T = 573.1; P = 0.46; L = 10; v = 30
    ug = 0.12; ud = 1.5

    R = 8.314
    t_step = 0.0005
    # contacting area
    # unit of P*1000 is MPa, unit of w is 'mm'
    w = (8*L/(np.pi*P*1000)) ** 0.5
    A = np.pi*w ** 2/4
    # density of PEG: unit 'mg/mm3'
    den_l = 1.125
    # wear coefficient in relation with temperature
    Kw = K*np.exp(-Qk/(R*T))
    # hardness of graphite(68.6-107.9MPa) average value in GPa
    H = 0.088

    t = np.zeros(5000)
    slidingdistance = np.zeros(5000)
    slidingdistance_tran = np.zeros(5000)
    at = np.zeros(5000)
    at_dot = np.zeros(5000)
    h2_dot = np.zeros(5000)
    h_liquid = np.zeros(5001)
    Hs = np.zeros(5000)
    hs_dot = np.zeros(5000)
    hs = np.zeros(5001)
    hg = np.zeros(5000)
    beta = np.zeros(5000)
    u = np.zeros(5000)

    # set initial thickness: unit 'mm'
    h_liquid[0] = h_l
    hs[0] = h_s

    for j in range(5000):
        t[j] = (j)*t_step

        slidingdistance_tran[j] = np.round(10000*v*t[j])
        slidingdistance[j] = slidingdistance_tran[j]/10000

        if h_liquid[j]>0:
            # instantaneous weight of PEG
            at[j] = h_liquid[j]*A*den_l

            # decrease of h_liquid due to degradation
            at_dot[j] = D*np.exp(-Qa/(R*T))*(at[j]) ** ka
            a_step = at_dot[j]*t_step
            h1_step = -a_step/(A*den_l)

            # decrease of h_liquid due to smear on the wear track
            h2_dot[j] = -h_liquid[j]*(c*(P ** k1)*(v ** k2))
            h2_step = h2_dot[j]*t_step

            # instantaneous h_liquid
            h_liquid[j+1] = h_liquid[j]+h1_step+h2_step
        else:
            h_liquid[j+1] = 0


        # calculation of graphite coating thickness
        if hs[j]>0:
            Hs[j] = H/(hs[j] ** k4)
            hs_dot[j] = Kw*(P ** k5*(v ** k3)/Hs[j])
            hs[j+1] = hs[j]-hs_dot[j]*t_step
        else:
            hs[j+1] = 0

        # calculation of cof
        hg[j] = h_liquid[j]+hs[j]
        beta[j] = np.exp(-(lambda1*(hg[j])*1000) ** lambda2)
        u[j] = (1-beta[j])*ug+beta[j]*ud

    plt.scatter(t, u)
    plt.show()

def run_liquid_model():
    lubricant_1 = Lubricant(1, 'ZEPF', True, False, density_liquid=1.125)
    variable_condition_1 =  VariableExperimentalConditions(450, 8, 0.34, 50, 25, 0)
    static_condition_1 = StaticExperimentalConditions(lubricant_1, 'P20', 0.3, 'AA7075', 0.7, 2, 'None', 0, 0)
    experiment_1 = Experiment(static_condition_1, variable_condition_1)

    temperature = 300; speed = 100; force = 5; pressure = 0.34; lubricant_thickness = 25
    density_liquid = 1.125

    lubricant_1 = Lubricant(1, 'ZEPF', True, False, density_liquid)

    condition_1 = StaticExperimentalConditions(lubricant=lubricant_1, pin_material='P20',
        pin_roughness=0.3, blank_material='AA7075', blank_roughness=0.7, blank_thickness=2,
        coating_material='None', coating_thickness=0, coating_roughness=0)

    parameters_1 = FrictionModelParametersLiquid(mu0_lubricated = 1.69073,
        Q_lubricated = 9141.50683, mu0_dry = 10.94225, Q_dry = 9368.85126,
        eta_0 = 0.12, Q_eta = 11930, lambda_1 = 40.70, lambda_2 = 1.55, c = 0.00847,
        k_1 = 1.52, k_2 = 2.67, k_3 = 4.58)

    friction_model_liquid = FrictionModelLiquid(condition_1, parameters_1)

    plotting_range = np.linspace(0, 100, num=420)
    prediction = friction_model_liquid.predict(plotting_range, temperature, force, pressure, speed, lubricant_thickness)

    plt.plot(plotting_range, prediction)
    plt.show()

def run_solid_liquid_model():
    # NOTE: Note that in this model lubricant thickness is in mm not micrometers
    temperature = 300; speed = 30; force = 10; pressure = 0.46; lubricant_thickness = 0.025; equiv_solid_thickness = 0.01
    density_liquid = 1.125; avg_hardness_solid = 0.088

    lubricant_2 = Lubricant(2, 'Omega 35', True, True, density_liquid, avg_hardness_solid)

    condition_2 = StaticExperimentalConditions(lubricant=lubricant_2, pin_material='P20',
        pin_roughness=0.3, blank_material='AA7075', blank_roughness=0.7,
        blank_thickness=2, coating_material='None', coating_thickness=0,
        coating_roughness=0)

    parameters_2 = FrictionModelParametersLiquidSolid(lambda_1=1.69633723376872, lambda_2=0.802696062360457,
        c=0.0130799271358491, k_1=0.801289664383808, k_2=2.07037498052891, D=2.66926336189041*10000, Q_alpha=4.98493351507642*10000, k_alpha=1.00674951577508,
        k_3=2.81418025418379, k_4=1.39617733322863, K=1.07360693295872*10000, Q_K=7.40524741745115*10000, k_5=1.85796986280561)

    friction_model_liquid_solid = FrictionModelLiquidSolid(condition_2, parameters_2)

    plotting_range = np.linspace(0, 100, num=420)

    t0 = time.perf_counter()
    prediction = friction_model_liquid_solid.predict(plotting_range, temperature, force, pressure, speed, lubricant_thickness, equiv_solid_thickness)
    print(time.perf_counter() - t0)

    plt.plot(plotting_range, prediction)
    plt.show()

    # import inspect
    # print(friction_model_liquid_solid.predict.__func__.__code__.co_varnames)
    # print(inspect.getfullargspec(friction_model_liquid_solid.predict).args)
    # print(inspect.getfullargspec(friction_model_liquid_solid.predict))

if __name__ == "__main__":
    # Ignore runtime warnings: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    run_liquid_model()
    run_solid_liquid_model()
