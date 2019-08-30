#!/usr/bin/env python3
"""Analytic Solution
====================
First, we want to have an idea as to what the analytic solution tells us.  The
problem is that the analytic equation that determines the energy eigenvalues
cannot be solved analytically; thus, numerical methods must be used to find the
energy eigenvalues of the system.
"""

import pathlib

import matplotlib.pyplot as pyplot
import numpy as np
from scipy.special import gamma




def energy(e: float, a_s: float) -> float:
    """The function that determines the energy eigenvalues.
    Parameters
    ==========
    e: float
      The input energy (in arbitrary units)
    a_s: float
      The chosen scattering length (in arbritrary units)
    Returns
    =======
    v: float
      Value of the scattering length - energy function evaluated for the given energy.
    """
    return gamma(-e/2 + 3/4) - gamma(-e/2 + 1/4) / (a_s * np.sqrt(2))


if __name__ == "__main__":
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # Example Plotting
    ########################################
    # Here is a simple example of a plotting routine.

    # First, we create an array of x values, and compute the corresponding y
    # values.
    x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
    y = np.sin(x) / x
    # We create the new figure with 1 subplot (the default), and store the
    # Figure and Axis object in `fig` and `ax` (allowing for their properties
    # to be changed).
    fig, ax = pyplot.subplots()
    ax.plot(x, y)
    ax.set_title("Plot of sin(x) / x")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Matplotlib by default does a good job of figuring out the limits of the
    # axes; however, it can fail sometimes.  This allows you to set them
    # manually.
    ax.set_ylim([-0.5, 1.1])
    fig.savefig("output/example.pdf")

    # Q3 _______________________________________________________________________
    # We want to find solutions to the scattering length (a_s) and energy relation
    # given in section 3. We will first hold the a_s constant, positive, "large".
    step = 100
    input_energy = np.linspace(0,20,step)
    print(input_energy)

    a_s = 10000

    output_error = np.array([energy(x, a_s) for x in input_energy])
    print(output_error)

    # Plotting Q3
    x = input_energy
    y = output_error
    # We create the new figure with 1 subplot (the default), and store the
    # Figure and Axis object in `fig` and `ax` (allowing for their properties
    # to be changed).
    fig, ax = pyplot.subplots()
    ax.plot(x, y)
    ax.set_title("Plot of error function vs. input energies")
    ax.set_xlabel("input_energy")
    ax.set_ylabel("output_error")
    ax.grid(linestyle = "-.", linewidth = 0.01)
    # Matplotlib by default does a good job of figuring out the limits of the
    # axes; however, it can fail sometimes.  This allows you to set them
    # manually.

    y_min = output_error.min()/(step) # Because energy is always positive
    y_max = output_error.max()/(step)
    ax.set_ylim([y_min, y_max])
    fig.savefig("output/Q3_plot_pretty.pdf")
