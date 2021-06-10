#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* FORCE MODULE *
Contains force calculations (and potential energies) using known potentials:
- Lennard-Jones

Latest update: June 1st 2021
"""

import numpy as np
import system
import matplotlib.pyplot as plt
from numba import jit, njit, vectorize

"""Parameters needed in the force calculations routines"""
epsilon = None
epsilon_wall = None
sigma = None
sigma_wall = None
cutoff = None
cutoff_wall = None
potential_shift = None


@njit
def mic(xi, xj, L):
    """Computes shortest distance based on minimum-image-convention (MIC).

    Args:
        xi -- position of i particle
        xj -- position of j particle
        L -- dimension of enclosing box

    Returns:
        rij -- the distance between particle i and j

    Note:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    rij = xi - xj
    if abs(rij) > 0.5 * L:
        rij = rij - np.sign(rij) * L

    return rij

@njit
def lennard_jones(force, pos, L):
    """Computes the force on all particles given a Lennard-Jones potential

    Args:
        force -- np.zeros([N, 3]) array for force storage (N = number of particles)
        pos -- np.array([N,3]) array of current positions
        L -- dimension of enclosing box

    Parameters:
        sigma -- parameter from the Lennard-Jones potential
        epsilon -- parameter from the Lennard-Jones potential

    Returns:
        force -- np.array([N, 3]) containing the net force on each particle
        potential -- value of the potential energy of the system

    Note:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    potential = 0
    rij = []

    for i in range(pos.shape[0] - 1):
        for j in range(i + 1, pos.shape[0]):

            r = 0

            #Generalized version --> probably very slow
            for dim in range(pos.shape[1]):
                dist = mic(pos[i,dim], pos[j,dim], L[dim])
                rij.append(dist)
                r += dist * dist

            if(r < cutoff**2):
                for dim in range(pos.shape[1]):
                    f = 48 * epsilon * (sigma**12 * rij[(-3+dim)] / r**7 - 0.5 * sigma**6 * rij[(-3+dim)] / r**4)
                    force[i, dim] += f
                    force[j, dim] -= f

                potential += 2 * 4 * epsilon * (sigma**12 / r**6 - sigma**6 / r**3) - potential_shift

    return force, potential

@njit
def lennard_jones_wall(pos, L, force, potential, axis = 2):
    """Computes contribution to the force on the particles due to the interactions
    with the wall using a 9-3 LJ potential

    Args:
        pos -- np.array([N,3]) array of current positions
        L -- dimension of enclosing box
        force -- np.array([N, 3]) containing forces (N = number of particles)
        potential -- value of potential energy of the system
        axis -- (default = 2) direction of the force, default is z-axis

    Parameters:
        sigma_wall -- parameter from the Lennard-Jones wall-interaction potential
        epsilon_wall -- parameter from the Lennard-Jones wall-interaction potential

    Returns:
        force -- updated array containing the net force of all particles
        potential -- updated value of the potential energy of the system

    Note:
        -- the routine runs using Numba @njit decorator for faster run time
        -- the routine is usually called after lennard_jones() -> force, potential already initialized
    """
    const = 3 * 0.5 * np.sqrt(3) * epsilon_wall * sigma_wall**3

    for i in range(pos.shape[0]):
        if(pos[i, axis] < cutoff_wall):
            z = pos[i, axis]
            potential += const * (sigma_wall**6 / z**9 - 1 / z**3)
            force[i, axis] -= const * (-9 * sigma_wall**6 / z**10 + 3 / z**4)


        if(pos[i, axis] > (L[axis] - cutoff_wall)):
            z = L[axis] - pos[i, axis]
            potential += const * (sigma_wall**6/z**9 - 1/z**3)
            force[i, axis] += const * (-9 * sigma_wall**6/z**10 + 3/z**4)


    return force, potential

@njit
def external_force(force, k, axis = 2):
    """Computes contribution to the force on the particles due to an external force
    directed on a given axis (default z-axis)

    Args:
        force -- np.array([N, 3]) containing forces
        k -- parameter for force strenght
        axis -- (default = 2) direction of the force, default is z-axis

    Parameters:
        sigma -- parameter from the Lennard-Jones potential
        epsilon -- parameter from the Lennard-Jones potential

    Returns:
        force -- updated array containing the net force of all particles
    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for i in range(force.shape[0]):
        force[i, axis] -= k * epsilon / sigma

    return force


def LJ_potential_shift():
    """Computes the LJ potential shift

    Parameters:
        cutoff -- radius at which the potential is truncated and shifted

    Notes:
        -- this is done once at the beginning of each simulation
    """
    global potential_shift
    potential_shift = 4*epsilon*(np.power(sigma/cutoff, 12) - np.power(sigma/cutoff, 6))
