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
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

"""Parameters needed in the force calculations routines"""
epsilon = None
epsilon_wall = None
sigma = None
sigma_wall = None
cutoff = None
cutoff_wall = None
potential_shift = None
kb = 0.00041
r0 = None

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


@njit(nogil=True)
def lennard_jones_parallel_istance(index, force, pos, L):
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
    virial = 0

    for i in range(index[0], index[1]):
        for j in range(i+1, pos.shape[0]):

            # X AXIS
            distx = pos[i,0] - pos[j,0]
            if abs(distx) > 0.5 * L[0]:
                distx = distx - np.sign(distx) * L[0]

            # Y AXIS
            disty = pos[i,1] - pos[j,1]
            if abs(disty) > 0.5 * L[1]:
                disty = disty - np.sign(disty) * L[1]

            # Z AXIS
            distz = pos[i,2] - pos[j,2]
            if abs(distz) > 0.5 * L[2]:
                distz = distz - np.sign(distz) * L[2]

            r = distx * distx + disty * disty + distz * distz


            if(r < cutoff**2):
                fx = 48 * epsilon * (sigma**12 * distx / r**7 - 0.5 * sigma**6 * distx / r**4)
                fy = 48 * epsilon * (sigma**12 * disty / r**7 - 0.5 * sigma**6 * disty / r**4)
                fz = 48 * epsilon * (sigma**12 * distz / r**7 - 0.5 * sigma**6 * distz / r**4)

                force[i, 0] += fx
                force[i, 1] += fy
                force[i, 2] += fz

                force[j, 0] -= fx
                force[j, 1] -= fy
                force[j, 2] -= fz

                virial += distx * fx
                virial += disty * fy
                virial += distz * fz

                potential += 2 * 4 * epsilon * (sigma**12 / r**6 - sigma**6 / r**3) - potential_shift

    return force, potential, virial


def lennard_jones(force, pos, L):
    """Parallelization of lennard_jones calculation over 4 cores
    """

    potential = 0
    virial = 0

    size = force.shape[0] // 4 + (force.shape[0] % 4 > 0)
    index = [np.array([0,size]),np.array([size, 2*size]),np.array([2*size, 3*size]),np.array([3*size, force.shape[0]-1])]


    with ThreadPoolExecutor(4) as ex:
        output = list(ex.map(lennard_jones_parallel_istance, index, repeat(force), repeat(pos), repeat(L)))


    pool1 = output[0]
    pool2 = output[1]
    pool3 = output[2]
    pool4 = output[3]

    force = pool1[0] + pool2[0] + pool3[0] + pool4[0]
    potential = pool1[1] + pool2[1] + pool3[1] + pool4[1]
    virial = pool1[2] + pool2[2] + pool3[2] + pool4[2]

    return force, potential, virial

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


@njit
def lennard_jones_bond(mask, force, pos, L):
    """Computes the force on all particles given a Lennard-Jones potential

    Args:
        force -- np.zeros([N, 3]) array for force storage (N = number of particles)
        pos -- np.array([N,3]) array of current positions
        L -- dimension of enclosing box
        charge -- np.array(N) containing particles charges

    Parameters:
        sigma -- parameter from the Lennard-Jones potential
        epsilon -- parameter from the Lennard-Jones potential

    Returns:
        force -- np.array([N, 3]) containing the net force on each particle
        potential -- value of the potential energy of the system

    Note:
        -- the routine runs using Numba @njit decorator for faster run time
    """

    for m in range(mask.shape[0]):
        for i in range(2):

            index = mask[m,i]
            if i == 0 : bonded = 1
            else : bonded = 0

            for j in range(pos.shape[0]):

                if(index != j and bonded != j):

                    # X AXIS
                    distx = pos[index,0] - pos[j,0]
                    if abs(distx) > 0.5 * L[0]:
                        distx = distx - np.sign(distx) * L[0]

                    # Y AXIS
                    disty = pos[index,1] - pos[j,1]
                    if abs(disty) > 0.5 * L[1]:
                        disty = disty - np.sign(disty) * L[1]

                    # Z AXIS
                    distz = pos[index,2] - pos[j,2]
                    if abs(distz) > 0.5 * L[2]:
                        distz = distz - np.sign(distz) * L[2]

                    r = distx * distx + disty * disty + distz * distz


                    if(np.sqrt(r) < cutoff):
                        fx = 48 * epsilon * (sigma**12 * distx / r**7 - 0.5 * sigma**6 * distx / r**4)
                        fy = 48 * epsilon * (sigma**12 * disty / r**7 - 0.5 * sigma**6 * disty / r**4)
                        fz = 48 * epsilon * (sigma**12 * distz / r**7 - 0.5 * sigma**6 * distz / r**4)

                        force[index, 0] += fx
                        force[index, 1] += fy
                        force[index, 2] += fz

                        force[j, 0] -= fx
                        force[j, 1] -= fy
                        force[j, 2] -= fz


    return force

@njit
def bond_potential(mask, pos, force):
    """Computes the bond potential between two atoms of the same molecule"""

    # loop over all molecules
    for m in range(mask.shape[0]):

        #compute bond forces between particles of given molecule
        i_index = mask[m,0]
        j_index = mask[m,1]

        rx = pos[i_index, 0] - pos[j_index, 0]
        ry = pos[i_index, 1] - pos[j_index, 1]
        rz = pos[i_index, 2] - pos[j_index, 2]

        r = np.sqrt(rx*rx + ry*ry + rz*rz)

        fx = kb*(r - r0)*rx/r
        fy = kb*(r - r0)*rx/r
        fz = kb*(r - r0)*rx/r

        force[i_index, 0] += fx
        force[i_index, 1] += fy
        force[i_index, 2] += fz

        force[j_index, 0] -= fx
        force[j_index, 1] -= fy
        force[j_index, 2] -= fz

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
