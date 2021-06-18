#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* ROUTINES MODULE *
Contains the majority of the routines that are not strictly related to
integration schemes and force calculation. These include:
- Density profile
- Radial distribution function
- (Copy) Minimum Image Convention (necessary for some routines)

Latest update: June 6th 2021
"""

import numpy as np
from numba import jit, njit, vectorize
import system
import integrator
import settings
import const
import force
import itertools
import matplotlib.pyplot as plt

"""Minimum image convention"""
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

"""Radial distribution function routines"""
@njit
def rdf_distances(pos, L, distances):
    """Computes distances between all particles needed for the radial
    distribution function calculations

    Args:
        pos -- np.array([N,3]) array of current positions
        L -- dimension of enclosing box
        distances -- np.zeros(N(N-1)) array for storing distances

    Returns:
        distances -- np.array(N(N-1)) containing distances squared

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """

    d_index = 0
    for i in range(pos.shape[0]):
        for j in range(pos.shape[0]):
            if i!=j:
                rx = mic(pos[i, 0],pos[j,0], L[0])
                ry = mic(pos[i, 1],pos[j,1], L[1])
                rz = mic(pos[i, 2],pos[j,2], L[2])
                distances[d_index] = np.sqrt(rx*rx + ry*ry + rz*rz)
                d_index += 1

    return distances

@njit
def rdf_normalize(norm, rdf, bins, N, rho):
    """Computes radial distribution function normalization

    Args:
        rdf -- np.array(nbins) containing frequencies
        bins -- np.array(nbins) containing bin values
        N -- total number of particles
        rho -- system number density

    Returns:
        rdf -- normalized rdf values

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for i in range(len(rdf)):
        shell_volume = 4*np.pi*(bins[i+1]**3 - bins[i]**3)/3
        norm[i] = rdf[i]/N/shell_volume/rho

    print(norm-rdf)
    return norm, bins[1:]

def rdf_fixed(bins, count, dist, N, rho):

    for i in range(len(bins)-1):
        for d in dist:
            if ((bins[i] <= d )&(d < bins[i+1])):
                count[i] += 1
         #normalize
        shell_volume = 4*np.pi*(bins[i+1]**3 - bins[i]**3)/3
        count[i] = count[i]/N/shell_volume/rho

    return count


def radial_distribution_function(nbins=200):
    """Computes the radial distribution function of the system, among with
    the coordination number and the isothermal compressibility

    Args:
        nbins -- (default = 100) number of sampling bins for the rdf histogram

    Returns:
        rdf -- np.array(nbins) with the radial distribution function values
        rdf_bins -- np.array(nbins+1) of bins
        n_c -- np.array(nbins) with the coordination number of the system
        k_t -- value of the isothermal compressibility
    """

    # Array of distances
    dist = np.zeros(system.N*(system.N-1))
    dist = rdf_distances(system.pos/force.sigma, system.L/force.sigma, dist)

    max_dist = 0.5*system.L[0]/force.sigma
    bins = np.linspace(0., max_dist, nbins)
    rdf = rdf_fixed(bins, np.zeros(len(bins)-1, dtype = np.float), dist, system.N, system.rho*force.sigma**3)

    # Radial Distribution Function
    #histogram = np.histogram(dist, bins=bins, density=False)
    #rdf, bins = rdf_normalize(np.zeros(histogram[0].shape, dtype=np.float), histogram[0], histogram[1], system.N, system.rho*force.sigma**3)



    # Coordination Number
    #n_c = 4*np.pi*system.rho * np.cumsum(rdf*bins[1]*bins[1:]**2)

    # Isothermal Compressibility
    #k_t = np.cumsum(4/(system.T) * np.pi * (rdf-1) * bins[1] * bins[1:]**2) + 1/(system.T  * system.rho)

    #return rdf, bins, 0, 0
    return rdf, bins[1:]

"""Density profile"""
def density_profile(axis, nbins = 100):
    """Computes the density profile of the particles over a given axis

    Args:
        axis -- specifies the axis along which to compute the density profile
        n_bins -- (default = 100) number of sampling bins for the histogram

    Returns:
        hist[0] -- histogram values -> y of density profile
        hist[1] -- histogram bins -> x of density profile
    """
    bins = np.linspace(0., system.L[axis], num=nbins)
    hist = np.histogram(system.pos[:,axis], bins=bins, density=True)
    return hist[0], hist[1]

"""Velocity routines: random, shift, rescale"""

def velocity_random(type = 'uniform'):
    """Randomly generates velocities in the range (-1,1)

    Args:
        type -- (default = 'uniform') type of distribution
    """
    if(type == 'uniform'):
        system.vel = np.random.uniform(-1.0, 1.0, (system.N,system.dim))
        #Velocities are shifted to avoid unwanted momenta
        for dim in range(system.vel.shape[1]):
            system.vel[:,dim] -= np.mean(system.vel[:,dim])

    elif(type == 'boltzmann'):
        sigma = system.T*const.KB/system.mass
        system.vel = np.sqrt(sigma)*np.random.normal(0, 1, size=(system.N, system.dim))

@njit
def v_res(v, Td, kb, mass):
    """Rescale velocities using numba

    Args:
        v -- np.array(N,dim) containing velocities
        Td -- desired temperature
        kb -- Boltzmann constant
        mass -- particles' mass

    Returns:
        v -- rescaled velocities
    """
    vel_sq = 0
    for axis in range(v.shape[1]):
        for i in range(v.shape[0]):
            vel_sq += v[i,axis]**2

    Tc = mass*vel_sq/3./kb/v.shape[0]
    factor = np.sqrt(Td/Tc)

    for axis in range(v.shape[1]):
        for i in range(v.shape[0]):
            v[i,axis] *= factor

    return v

def velocity_rescale():
    """Calls Numba for rescaling the velocities
    """
    system.vel = v_res(system.vel, system.T, const.KB, system.mass)

"""Position routines: distribute lattice, get box dimensions"""
def position_lattice(type = 'cubic'):
    """Distributes position over a lattice given the type

    Args:
        type -- (default = 'cubic') type of lattice: cubic or rectangular-z

    Errors:
        -- unspecified lattice type
    """

    x = np.zeros(system.N)
    y = np.zeros(system.N)
    z = np.zeros(system.N)

    places_z = np.linspace(0, system.L[2], num=system.n[2], endpoint = False)
    places_z += places_z[1]*0.5
    n_part = 0

    for i,j,k in itertools.product(list(np.arange(system.n[0])), list(np.arange(system.n[1])), list(np.arange(system.n[2]))):

        x[n_part] = system.alat*i
        y[n_part] = system.alat*j

        if(type=='cubic'):
            z[n_part] = system.alat*k
        elif(type == 'rectangular-z'):
            z[n_part] = places_z[k]
        else:
            print("Error: unknown lattice type.")
        n_part += 1

    system.pos[:,0] = x
    system.pos[:,1] = y
    system.pos[:,2] = z

def get_box_dimensions():

    L = np.zeros(3)
    l = np.cbrt(system.N/system.rho)
    L[0] = l
    L[1] = l
    L[2] = l

    return L

"""Energy routines: compute energies, compute temperatures"""

@njit
def nose_hoover_energy(Q, xi, N, kb, T, lns):
    """Computes the Nose Hoover energy contribution given by

    E = xi*xi*Q/2 + 3NkbTlns
    """
    energy = 0.5*Q*xi**2 + 3*N*kb*T*lns
    return energy


"""Nose-Hoover specific routines"""
@njit
def compute_G(kinetic, N, kb, T, Q):
    """Computes the variable G
    """
    G = (2*kinetic - 3*N*kb*T)/Q
    return G

def compute_Q():
    """Computes the thermal mass (once per simulation)
    """
    g = 3*system.N + 1
    system.Q = g*const.KB*system.T*settings.DT**2*settings.m**2


"""Statistical and fluctuations routine: std deviation, variance, running average"""
def statistical(array):
    """Computes mean, standard deviation, and variance

    Args:
        array -- np.array containing the desired quantity

    Returns:
        [mean,std,var] -- standard deviation, variance
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    mean = np.mean(array)
    std = np.std(array)
    var = np.var(array)

    return [mean,std,var]

def running_average(array, dt):
    """Computes the running average of a given observable at time t

    Args:
        array -- input observable array
        dt -- timestep

    Returns:
        r_ave -- running average
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    r_ave = np.cumsum(array*dt)
    for j in range(len(r_ave)):
        r_ave = r_ave/(dt*(j+1))
    return r_ave

"""Thermodynamical routines"""
def specific_heat(energy, potential, temperature):
    """Computes specific heat of the system using two different methods:
    (1) total energy variance and (2) potential energy variance. The temperature is given by
    the mean of all values up to current time.

    Args:
        energy -- list containing total energy for multiple timesteps
        potential -- list containing potential energy for multiple timesteps
        temperature -- list containing computed temperature for multiple timesteps

    Returns:
        CV[0] -- specific heat computed with (1)
        CV[1] -- specific heat computed with (2)
    """

    var_e = statistical(np.asarray(energy))[2]
    var_u = statistical(np.asarray(potential))[2]
    mean_t = statistical(np.asarray(temperature))[0]

    CV_e = var_e/const.KB/mean_t**2
    CV_u = (1/const.KB/mean_t**2)*(var_u + 0.5*3*system.N*(mean_t*const.KB)**2)

    system.cv = [CV_e, CV_u]

    return [CV_e, CV_u]

def current_temp():
    """Computes current temperature using the kinetic energy relation
    """
    temp = 2*system.kinetic/3./const.KB/system.N
    return temp
