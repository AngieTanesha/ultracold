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
import numpy.ma as ma




def energy(e: float) -> float:
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
    return  gamma(-e/2 + 1/4) / (gamma(-e/2 + 3/4) * np.sqrt(2))


if __name__ == "__main__":
    # Make sure that the output/ directory exists, or create it otherwise.
    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()


    # Q3 _______________________________________________________________________
    # We want to find solutions to the scattering length (a_s) and energy relation
    # given in section 3. We will first hold the a_s constant, positive, "large".
    step = 10000
    input_energy = np.linspace(-10,20,step)
    print(input_energy)


    a_s = np.array([energy(x) for x in input_energy])
    print(a_s)

    # Plotting Q3
    x = a_s
    y = input_energy
    x = ma.masked_where(abs(x)>50, x)
    # We create the new figure with 1 subplot (the default), and store the
    # Figure and Axis object in `fig` and `ax` (allowing for their properties
    # to be changed).
    fig, ax = pyplot.subplots()
    ax.plot(x, y)
    ax.set_title("Plot of input energy ($E_{input}$) vs. scattering length ($a_s$)")
    ax.set_xlabel("$a_s$")
    ax.set_ylabel("$E_{input}$")
    ax.grid(linestyle = "-.", linewidth = 0.01)
    # Matplotlib by default does a good job of figuring out the limits of the
    # axes; however, it can fail sometimes.  This allows you to set them
    # manually.

    ax.set_ylim(-5,15)
    ax.set_xlim(-5,5)
    fig.savefig("output/Q3_plot_pretty3_use this.pdf")
