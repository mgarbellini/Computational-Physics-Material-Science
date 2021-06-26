#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* FORCE MODULE *
Contains force calculations (and potential energies) using known potentials.

Latest update: June 25th 2021
"""
import system
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import special
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
Rc = None
c_shift = None
potential_shift = None
epsilon_not = 8.9875517923E9


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

@njit
def lennard_jones_serial_istance(force, pos, L, charge):
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
    potential = 0
    virial = 0

    for i in range(pos.shape[0]):
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

            """Lennard Jones Potential Cutoff"""
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

            """Coulomb Potential Cutoff"""
            if(r < Rc**2):

                k = charge[i]*charge[j]/4/np.pi/epsilon_not
                rij = np.sqrt(r)

                fx = k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*distx/r + c_shift*distx/rij)
                fy = k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*disty/r + c_shift*disty/rij)
                fz = k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*distz/r + c_shift*distz/rij)

                force[i, 0] += fx
                force[i, 1] += fy
                force[i, 2] += fz

                force[j, 0] -= fx
                force[j, 1] -= fy
                force[j, 2] -= fz


                potential += 2*k*(math.erfc(rij/Rc)/rij - math.erfc(1)/Rc + c_shift*(rij - Rc))




    return force, potential, virial


def lennard_jones(force, pos, L, parallel = False):
    """Parallelization of lennard_jones calculation over 4 cores
    """

    potential = 0
    virial = 0

    if parallel == True:

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

    else:

        force, potential, virial = lennard_jones_serial_istance(force, pos, L, system.charge)

    return force, potential, virial

@njit
def lennard_jones_wall(force, pos, L,potential, axis = 2):
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
def coulombic_wall(force, pos, pos_discrete, charge, discrete_surface_q, L):
    """Computes electrostatic force contribution due to the interactions of the particles
    inside the box with the charges on the charged surfaces
    """

    """Surface charges at z = 0"""
    surface = 0
    for i in range(pos.shape[0]):
        for j in range(pos_discrete.shape[0]):

            distx = pos[i,0] - pos_discrete[j,0, surface]
            if abs(distx) > 0.5 * L[0]:
                distx = distx - np.sign(distx) * L[0]

            disty = pos[i,1] - pos_discrete[j,1, surface]
            if abs(disty) > 0.5 * L[1]:
                disty = disty - np.sign(disty) * L[1]

            distz = pos[i,2] - 0.
            r = distx * distx + disty * disty + distz * distz
            k = charge[i]*discrete_surface_q/4/np.pi/epsilon_not
            rij = np.sqrt(r)

            if rij < Rc:
                force[i,0] += k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*distx/r + c_shift*distx/rij)
                force[i,1] += k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*disty/r + c_shift*disty/rij)
                force[i,2] += k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*distz/r + c_shift*distz/rij)

    """Surface charges at z = Lz """
    surface = 1
    for i in range(pos.shape[0]):
        for j in range(pos_discrete.shape[0]):

            distx = pos[i,0] - pos_discrete[j,0, surface]
            if abs(distx) > 0.5 * L[0]:
                distx = distx - np.sign(distx) * L[0]

            disty = pos[i,1] - pos_discrete[j,1, surface]
            if abs(disty) > 0.5 * L[1]:
                disty = disty - np.sign(disty) * L[1]

            distz = pos[i,2] - L[2]
            r = distx * distx + disty * disty + distz * distz

            k = charge[i]*discrete_surface_q/4/np.pi/epsilon_not
            rij = np.sqrt(r)

            if rij < Rc:
                force[i,0] -= k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*distx/r + c_shift*distx/rij)
                force[i,1] -= k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*disty/r + c_shift*disty/rij)
                force[i,2] -= k * (-2*np.exp(-(rij/Rc)**2)/rij/Rc/np.sqrt(np.pi) -math.erfc(rij/Rc)*distz/r + c_shift*distz/rij)

    return force

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
