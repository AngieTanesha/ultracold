#!/usr/bin/env python3
"""Numerical Solution
=====================
I changed sth
Now that we have an analytic solution, we can use the stochastic variational
method to get a numerical estimate and check whether it matches what we expect.
The idea of stochastic methods is that it uses some sort of randomness.  In
this instance, we want to construct a basis of Gaussian functions, but have no
good way of determining whether a Gaussian function is a good candidate without
doing the full calculation.  Because of this, we might as well just try a
number of random Gaussian functions and only keep the best ones to build the
basis.
"""

import pathlib

import matplotlib.pyplot as pyplot
import numpy as np
import scipy as sp
import scipy.special
import math as m

from analytic_solution import gamma

import r0v0


def gaussian_integral(k: float, n: float) -> float:
    """Computes exactly the Gaussian integral of the form:
    > ∫ rⁿ exp(-k x^2) dr
    where n > -1, k ∈ ℝ and the integral is done over r ∈ [0, ∞).
    We know this is equivalent to [1 / (2 * sqrt(a^(n + 1)))] * gamma((n + 1) / 2)
    Parameters
    ==========
    n: float
      The exponent of the integration variable outside of the exponential
    k: float
      The constant in the exponential
    Returns
    =======
    integral: float
      The value of the integral
    """
    return 1 / (2 * np.sqrt(k ** (n+1) )) * gamma((n + 1) / 2)


def eigenvalues(S, H):
    """Calculate the energy eigenvalues associated with the matrices S and H.
    If the generalized matrix equation is numerically unstable, an error will
    be raised.  This is completely normal and is to be expected.  You can
    `catch` the error and handle as appropriate with the following:
    ```
    try:
        # Code that might cause an error
    except ErrorName:
        # How to proceeded with the error
    ```
    Parameters
    ==========
    S: array-like, shape (N, N)
    H: array-like, shape (N, N)
        The S and H matrices for which to calculate the eigenvalues.
    Returns
    =======
    v: array-like, shape (N,)
        The list of eigenvalues, given in increasing order.
    """
    with np.errstate(all='raise'):
        # eigh returns two things: the list of eigenvalues and the list of
        # corresponding eigenvectors.  We don't really care about the
        # eigenvectors so we just return the eigenvalues.
        try:
            sp.linalg.eigh(H, S)[0]

        except (np.linalg.linalg.LinAlgError, ValueError):
            return False

        return sp.linalg.eigh(H, S)[0]


def lowest_eigenvalue(S, H):
    """Calculate the lowest (energy) eigenvalues associated with the matrices S
    and H.
    Parameters
    ==========
    S: array-like, shape (N, N)
    H: array-like, shape (N, N)
        The S and H matrices for which to calculate the eigenvalues.
    Returns
    =======
    The lowest eigenvalue and its index of the generalized matrix equation.
    """
    evalues = eigenvalues(S,H)

    if np.any(evalues):

        # Reject if ground state energies are less than zero.
        if np.amin(evalues) < 0:

            return False

        else:

            # Returning all eigenvalues, but this increases by one each time.
            return evalues

    else:
        return False

def plot_fit(x, y, p, Cp, nsig=1, ax=None):
    """Plot data along with the polynomial fit's confidence interval.
    See the example below for a concrete usage example.
    Parameters
    ==========
    x : array_like, shape (N,)
      The x values of the points used for the fit
    y : array_like, shape (N,)
      The y values of the points used for the fit
    p : array_like, shape (d,)
      The array of coefficients returned by `polyfit`, where `d` is the degree
      of the polynomial.
    Cp : array_like, shape (d, d)
      The covariance matrix returned by `polyfit`, where `d` is the degree of
      the polynomial.
    nsig : flo[0][0]at, default 1
      The number of sigmas to display
    ax : Pyplot Axis, default None
      The axis in which
    Returns
    =======
    The axis used for plotting which were newly created if `ax` was left None,
    or are the same as the one specific in the arguments.
    Examples
    ========
    ```
    x = np.linspace(0, 5, 10)
    y = 3*x + 10*x*x - 4
    y += np.randn(10)
    p, Cp = np.polyfit(x, y, deg=2, cov=True)
    plot_fit(x, y, p, Cp)
    '''
    """
    if ax is None:
        _, ax = pyplot.subplots()

    ax.plot(x, y, '.')

    margin = 0.1 * (x.max() - x.min())
    fit_x = np.linspace(x.min() - margin, x.max() + margin, 500)
    fit_y, y_sig = polyfit_error(fit_x, p, Cp)

    ax.fill_between(
        fit_x, fit_y + nsig * y_sig, fit_y - nsig * y_sig, alpha=0.1)

    return ax


def polyfit_error(x, p, Cp):
    """Given a polynomial fit with its covariance matrix, returns the best fit
    value at `x` along with the standard deviation.
    Parameters
    ==========
    x : array_like, shape (N,)
      The x values of the points used for the fit
    p : array_like, shape (d,)
      The array of coefficients returned by `polyfit`, where `d` is the degree
      of the polynomial.
    Cp : array_like, shape (d, d)
      The covariance matrix returned by `polyfit`, where `d` is the degree of
      the polynomial.
    Returns
    =======
    The pair `(y, y_sig)` where `y` is the best fit, and `y_sig` is the
    standard deviation on y.  Both y and y_sig are the same length as `x`.
    """
    deg = len(p) - 1
    TT = np.vstack([x**(deg - i) for i in range(deg + 1)]).T
    fit_y = np.dot(TT, p)
    y_cov = np.dot(TT, np.dot(Cp, TT.T))
    y_sig = np.sqrt(np.diag(y_cov))
    return fit_y, y_sig

def plot_gamma_functions(max_k: int):
    # __________________________________________________________________________
    # Testing the gaussian integral is a gamma function

    fig, ax = pyplot.subplots()

    # Plotting Q3
    for k in range(0,2*max_k,2):
        x = np.linspace(0,100,500) # These are n
        y = np.array([gaussian_integral(k, x_i) for x_i in x])
        #y = ma.masked_where(abs(y)>2000, y)
        ax.plot(x,y, label=f"k = {k}")

    ax.set_title("Plot of integral of a gaussian of k and n")
    ax.set_xlabel("$n$")
    ax.set_ylabel("$integral$")
    ax.grid(linestyle = "-.", linewidth = 0.01)
    ax.legend()
    ax.set_xlim(0,100)
    ax.set_ylim(0,2000)
    #pyplot.show()

    fig.savefig("output/numerical_gaussian_test_zoomed.pdf")

    return None


class System():
    """The system class in which all the parameters will be stored.

    The system has some parameters (such as `r0` and `v0` which determine the
    trapping term) which are used throughout.  Although they are constant for a
    particular system, they in general can change between different systems and
    thus cannot be hardcoded.  This means that we would have to pass `r0` and
    `v0` into all the different functions all the time which quickly becomes
    cumbersome.  Similarly, we will sometimes want to have access to the S and
    H matrices, but it is also cumbersome having to specify them multiple
    times.
    For this reason, we create a class and store `r0` and `v0` inside the
    class.  Functions within the class, which are called methods, behave in
    much the same way except that they always take one extra argument at the
    start called `self`.  This `self` argument is special and allows for `r0`
    and `v0` to be accessed through `self.r0` and `self.v0`.
    Because of this special `self` argument, the declaration a method is:
    ```
    def foo(self, x, y):
        # ...
    ```
    and a call to the function within the class is done through `self.foo(x,
    y)`.
    Ultimately, the class will need to be created with definite values which is
    done through:
    ```
    system = System(r0=5, v0=10)
    ```raise NotImplementedError()
    With this instance of the class, the variables within the class and methods
    of the class can be accessed as follows:
    ```
    potential_depth = system.v0
    system.foo(x ,y)
    ```
    """

    def __init__(
            self,
            v0: float,
            r0: float,
            mass=1,
            omega=1,
    ):
        """Instantiate a new System with the given parameters.
        This is a special function, hence the weird name with the underscores
        and has already been implemented for you.  In particular, this function
        is called when you first create a new instance of the `System` class.
        Notice that this creates all the different variables stored within the
        class.  Later on in other methods, you can access the S matrix by
        through `self.S`, and the states through `self.states`, etc.
        """
        # Keep track of what states we have in our basis, and the associate S
        # and H matrices.  Since we're nearly always dealing with the
        # inverse-square of the widths of the Gaussian, we don't store the
        # actual width in `self.states` but store the inverse-square of the
        # widths.
        self.states = np.zeros(0)
        self.energies = []

        self.S = np.zeros((self.states.shape[0], self.states.shape[0]))
        self.H = np.zeros((self.states.shape[0], self.states.shape[0]))

        # Store parameters global to the system
        self.v0 = v0
        self.r0 = r0

        # We store also mass and omega in case the student wishes to
        # investigate the effect what happens if mass and/or omega are changed.
        self.mass = mass
        self.omega = omega
        self.omega_squared = omega * omega

    def calculate_S_elem(self, si: float, sj: float) -> float:
        """Compute a particular matrix element S_{ij} with the given state `si`
        and `sj`.
        Note that this function should *not* update self.S and instead returns
        the matrix element.
        Parameters
        ==========
        si: float
        sj: float
          The inverse square of the widths for both states.  This does *not*
          take the indices of the states.
        Returns
        =======
        Sij: float
          The S matrix element associated with the two state.
        """
        return gaussian_integral(0.5 * (si + sj), 2)

    def calculate_S(self, new_state=None):
        """Compute the matrix S_{ij} from scratch and return it.
        This should use all of the states stored in `self.states` in order to
        compute the H matrix.  If a `new_state` is also specified (i.e. is not
        `None`), it should be included to compute the last row and column of
        the matrix.
        The function does *not* update `self.S` since we don't necessarily know
        whether `new_state` will be added to `self.states`.
        Parameters
        ==========
        new_state: float, default None
          If not None, also include `new_state` as a state when computing the S
          matrix.
        Returns
        =======
        The S matrix.
        """

        if np.any(new_state):

            # If there are new states, and new_state has to be a 1 x n array
            # (it is important for new_state to be a Numpy array for thelowest
            # following code to work).

            # Combine the states from the system with the new state
            states = np.append(self.states, new_state)

            # Create a row and column vector from this list of new states
            si, sj = np.meshgrid(states, states)

            # Return S elements as an n x n array
            return self.calculate_S_elem(si, sj)


        else:
            # If there are no new states, return the S elements using the states
            # that are already in the system.

            si, sj = np.meshgrid(self.states, self.states)

            return self.calculate_S_elem(si, sj)



    def calculate_H_elem(self, si: float, sj: float) -> float:
        """Compute a particular matrix element H_{ij} with the given state `si`
        and `sj`.  This does *not* store the value in the matrix.
        Parameters
        ==========
        si: float
        sj: float
          The inverse square of the widths for both states.  This does *not*
          take the indices of the states.
        Returns
        -------
        The H matrix element associated with the two state.
        """
        k = 0.5 * (si + sj)
        k2 = 0.5 * (1 / (self.r0**2) + si + sj)
        return (-0.5 * (sj ** 2) + 0.5) * gaussian_integral(k, 4) + 1.5 * sj * gaussian_integral(k, 2) + self.v0 * gaussian_integral(k2,2)

    def calculate_H(self, new_state=None):
        """Compute the matrix H_{ij} from scratch and return it.
        This should use all of the states stored in `self.states` in order to
        compute the H matrix.  If a `new_state` is also specified (i.e. is not
        `None`), it should be included to compute the last row and column of
        the matrix.
        The function does *not* update `self.H` since we don't necessarily know
        whether `new_state` will be added to `self.states`.
        Parameters
        ==========
        new_state: float, default None
          If not None, also include `new_state` as a state when computing the S
          matrix.
        Returns
        =======
        The H matrix.
        """

        if np.any(new_state):

            # If there are new states, and new_state has to be a 1 x n array
            # (it is important for new_state to be a Numpy array for the
            # following code to work).

            # Combine the states from the system print("S \n", S)with the new state
            states = np.append(self.states, new_state)

            # Create a row and column vector from this list of new states
            si, sj = np.meshgrid(states, states)

            # Return S elements as an n x n array
            return self.calculate_H_elem(si, sj)


        else:
            # If there are no new states, return the H elements using the states
            # that are already in the system.

            si, sj = np.meshgrid(self.states, self.states)

            return self.calculate_H_elem(si, sj)

    def gen_random_state(self, size=1) -> float:
        """Generate a random state with width between [r0 / 100, 100 * r0).
        The state is *not* added to the internal list of states.
        A list of random states can be created at once by specifying a `size`.
        For example `gen_random_state(5)` will create a list of 5 random
        states.
        Parameters
        ==========
        size: int, optional
          Specify the number of random states the generate.  If None, only a
          single random state is create.
        Returns
        =======
        The random state, or array of random states with the specified size.
        """
        return np.power(np.random.uniform(self.r0 / 100, 100 * self.r0, size), -2)


    def find_new_state(self, tries=2048):
        """Given the current states in the system, find a new state that
        improves the fit to the ground state.
        The states will contain a basis of `[s1, s2, ..., sN]`, and we want to
        find a new state such that, when appended to the current basis, will
        provided the best improvement the fit to the ground state.
        Note that since we will be trialling random states, there is no
        guarantee that this function finds an improvement; thus you must take
        care of handling both the scenarios in which an improvement is found,
        and in which no improvement is found.

        Parameters
        ==========
        tries: int
          The number of states to randomly generate before return the best
          candidate.

        Returns
        =======
        You decide (though probably should return the best state).
        """

        # First create the S and H matrices from the states that we already have.
        S_ori = self.calculate_S()
        H_ori = self.calculate_H()

        # Calculate the lowest eigenvalue
        best_e = lowest_eigenvalue(S_ori, H_ori)

        best_state = None

        for state in self.gen_random_state(tries):
            # Create the new S and H matrix by only considering one new state
            # and the old states at a time.
            S = self.calculate_S(state)
            H = self.calculate_H(state)

            #print("S  \n", S)
            #print("H  \n", H)

            # Get the new energy eigenvalue and its index considering the new state
            e = lowest_eigenvalue(S, H)

            # If there are errors, return False immediately
            if not np.any(e):
                continue

            #print(evalue, index)

            # Update the minimum eigenvalues and its index a smaller number is found.
            #print(f"{evalue} - {min_evalue} = abs({evalue-min_evalue})")
            #print(f"min evalue {min_evalue}")


            # Only compare GROUND STATE energies!
            if e[0] < best_e[0]:
                best_e = e
                best_state = state

        return best_state, best_e

    def construct_basis(self, nstates: int, max_failures=16):
        """This construct the basis with the desired number of states.
        This is the main loop with will try and construct a basis with the
        desired number of states, `nstates`.  Note that this is a stochastic
        process and in particular, you have to handle the possibility that new
        states or improvements can't be found.
        This function should update `self.states` as it goes.

        Parameters
        ==========

        nstates: int
          The desired number of states

        max_failures: int
          The maximum number of failures permitted when trying to find a new
          state.

        Returns
        =======
        Up to you to decide what, if anything, this function should return.
        """

        # start with no basis
        nbasis = 0
        nfails = 0

        # Generate one random state
        if self.states.shape[0] == 0:

            self.states = self.gen_random_state()
            S = self.calculate_S()
            H = self.calculate_H()
            self.energies = lowest_eigenvalue(S,H)

            nbasis += 1
            #print(f"    nbasis: {nbasis} with states {self.states}")

        while nbasis < nstates and nfails < max_failures:

            # new_e will be a different list each time.
            new_basis, new_e = self.find_new_state()

            if new_basis is None:
                nfails += 1

                if nfails == 15:
                    print(f"        {nfails} failures for basis {nbasis}, {self.states[nbasis-1]}")

                continue

            nbasis += 1
            self.states = np.append(self.states, new_basis)
            self.energies = new_e

            if nfails:
                print(f"        {nfails} failures for basis {nbasis}, {self.states[nbasis-1]}")

        return self.states, self.energies



if __name__ == "__main__":
    # If you are going to do a polynomial fit, have a loop at the documentation
    # for NumPy's `polyfit` function.  You can also have a look at the
    # functions provided at the top of this file called `plot_fit` and
    # `polyfit_error` which will assist in using the result from NumPy's
    # polyfit.

    # __________________________________________________________________________
    # Make sure that the output/ directory exists, or create it otherwise.

    output_dir = pathlib.Path.cwd() / "output"
    if not output_dir.is_dir():
        output_dir.mkdir()

    # See what the gamma function look like as k and n varies
    plot_gamma_functions(5)

    # __________________________________________________________________________

    # Constructing the basis and getting their ground state energies

    # Initialise the system, ie. their r0 and v0 values
    system = System(r0=r0v0.R0[0], v0=r0v0.V0[0])

    # Set the desired number of basis as num_basis
    num_basis = int(input("Enter the desired number of basis: "))
    #energy = int(input("Which excited state? "))
    filename = input("Enter filename: ")

    print(f"\nCreating {num_basis} basis")
    print("=========================")

    # print the number of errors and the basis this algorithm gets when calculating the eigenvalues.
    print(f"Errors (if any): \n")

    # __________________________________________________________________________
    # # ONE SET OF GROUND ENERGIES

    #
    # Get the basis and their ground state energies.
    # basis, ground = system.construct_basis(num_basis)
    #
    # # Print the basis and their energies.
    # print("=========================")
    # print(f"Final state of {len(basis)} basis: \n")
    # print(basis)
    #
    # print(f"\nwith ground energies\n")
    # print(ground)
    #
    # Plotting one set of ground state energy and the basis size
    #
    # x = np.arange(len(basis))
    # y = ground
    #
    # fig, ax = pyplot.subplots()
    # ax.plot(x, y)
    # ax.set_title("Plot of ground state energies vs. size of basis")
    # ax.set_xlabel("size of basis")
    # ax.set_ylabel("$E_0$")
    # ax.grid(linestyle = "-.", linewidth = 0.01)
    #
    # #ax.set_xlim(-10,0)
    # #ax.set_ylim(-1000,1000)
    # #pyplot.show()
    #
    # print("\n=========================")
    # filename = input("Enter filename: ")
    #
    # fig.savefig("output/Q7_groundstate_" + str(len(basis)) + "_" + str(filename) +".pdf")

    # __________________________________________________________________________
    # MULTIPLE GROUND ENERGIES
    # Plotting multiple ground state energies and basis size
    #
    # Get the basis and their ground state energies.

    fig, ax = pyplot.subplots()
    ax.set_yscale("log")

    avg_evalues = []
    all_evalues = []

    # Get all eigenvalues
    for i in range(10):
        print(f"  for plot {i} =========================\n")
        basis, evalues = system.construct_basis(num_basis)

        # Convert lists to numpy
        evalues = np.array(evalues)

        if i == 0:
            avg_evalues = np.zeros(len(evalues))
            all_evalues = evalues
            ax.plot(all_evalues, lw=0.5)
            system.states = np.zeros(0)
            system.energies = np.zeros(0)
            continue

        all_evalues = np.vstack((all_evalues, evalues))

        avg_evalues = (avg_evalues + evalues)/2

        ax.plot(all_evalues[i], lw=0.05)

        # Wipe all states and energies for the next construct basis call
        system.states = np.zeros(0)
        system.energies = np.zeros(0)


    ax.plot(avg_evalues,color="black", label="Average", lw = 0.8)

    ax.legend()
    ax.set_title("Plot of " +  filename + " state energies")
    ax.set_xlabel("Excited state n")
    ax.set_ylabel("$E_n$")
    ax.grid(linestyle = "-.", linewidth = 0.01)

    fig.savefig("output/Q7_energy_average_log_" + str(num_basis) + "_" + str(filename) +".pdf")
