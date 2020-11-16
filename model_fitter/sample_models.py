import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def gauss(x, amp, cen, sig):
    """Gaussian lineshape."""
    print('passed')
    return amp * np.exp(-(x-cen)**2 / (2.*sig**2))

class GaussModel():

    def __init__(self, parameters, conditions):
        # These are static conditions
        self.parameters = parameters
        self.conditions = conditions
        self.model_name = "Gaussian model"

    def plot_results(self, plotting_range, temperature, force,
        pressure, speed, lubricant_thickness):

        y_pred = self.predict(plotting_range, temperature,
            force, pressure, speed, lubricant_thickness)

        _, ax = plt.subplots()
        ax.plot(plotting_range, y_pred)
        plt.show()

    def predict(self, plotting_range, temperature,
            force, pressure, speed, lubricant_thickness):

        return self.parameters.amp * np.exp(-(plotting_range-self.parameters.cen)**2 / (2.*self.parameters.sig**2)) + speed


class GaussParameters():

    def __init__(self, amp, cen, sig):
        self.amp = amp
        self.cen = cen
        self.sig = sig


class GaussStaticConditions():

    def __init__(self, pin_material, pin_roughness, blank_material, blank_roughness):
        self.pin_material = pin_material
        self.pin_roughness = pin_roughness
        self.blank_material = blank_material
        self.blank_roughness = blank_roughness


class Data():

    def __init__(self, x_data, y_data) -> None:
        self.x_data = x_data
        self.y_data = y_data


class GaussVariableConditionsData(Data):

    def __init__(self, temperature, force, pressure, speed, lubricant_thickness, x_data, y_data) -> None:
        super().__init__(x_data, y_data)
        # In this case x_data will all be the same length. The same with y_data
        self.temperature = temperature
        self.force = force
        self.pressure = pressure
        self.speed = speed
        self.lubricant_thickness = lubricant_thickness

