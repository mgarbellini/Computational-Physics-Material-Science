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

Latest update: May 27th 2021
"""

import numpy as np
from numba import jit, njit, vectorize
import system
import itertools

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
	for i in range(pos.shape[0] - 1):
		for j in range(i + 1, pos.shape[0]):
			rx = mic(pos[i, 0],pos[j,0], L[0])
			ry = mic(pos[i, 1],pos[j,1], L[1])
			rz = mic(pos[i, 2],pos[j,2], L[2])
			distances[d_index] = rx*rx + ry*ry + rz*rz
			distances[d_index+1] = distances[d_index]

	return distances

@njit
def rdf_normalize(rdf, bins, N, rho):
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
	for i in range(rdf.shape[0]):
		shell_volume = 4*np.pi*((bins[i+1])**3-bins[i]**3)/3
		rdf[i] = rdf[i]/(N*shell_volume*rho)

	return rdf


def radial_distribution_function(nbins=100):
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
	dist = rdf_distances(system.pos, system.L, dist)
	dist = np.sqrt(dist)

	max_dist = 0.5*system.L[0]
	bins = np.linspace(0., max_dist, nbins)

	# Radial Distribution Function
	histogram = np.histogram(dist, bins=bins, density=False)
	rdf = rdf_normalize(histogram[0], histogram[1], system.N, system.rho)
	rdf_bins = histogram[1]


	# Coordination Number
	n_c = 4*np.pi*system.rho * np.cumsum(rdf*bins[1]*bins[1:]**2)

	# Isothermal Compressibility
	k_t = np.cumsum(4/(system.T) * np.pi * (rdf-1) * bins[1] * bins[1:]**2) + 1/(system.T  * system.rho)

	return rdf, rdf_bins, n_c, k_t[-1]

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
def vel_random(type = 'uniform'):
    """Randomly generates velocities in the range (-1,1)

    Args:
        type -- (default = 'uniform') type of distribution
    """
    if(type == 'uniform'):
        system.vel = np.random.uniform(-1.0, 1.0, (system.N,system.dim))
    elif(type == 'boltzmann'):
        print("yet to be implemented")

@njit
def velocity_shift(vel):
    """Shifts the velocities to cancel any unwanted linear momentum

    Args:
        vel -- np.array(N,dim) containing current velocities

    Returns:
        vel -- array with shifted velocities

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """

    for dim in range(vel.shape[1]):
        mean = 0
        for i in range(vel.shape[0]):
            mean += vel[i,dim]
        mean = mean/vel.shape[0]
        for i in range(vel.shape[0]):
            vel[i,dim] -= mean

    return vel

def vel_shift():
    """Shifts the velocities to cancel any unwanted linear momentum
    """
    mean = np.sum(system.vel, axis = 0)/system.N
    system.vel[:,0] -= mean[0]
    system.vel[:,1] -= mean[1]
    if (dim == 3):
        system.vel[:,2] -= mean[2]

def vel_rescale(temp):
    """Rescales the particle velocities to a target temperature

    Args:
        temp -- value of temperature at which the velocities are rescaled
    """
    v_ave = np.sum(np.multiply(system.vel, system.vel))/system.N
    system.vel *= np.sqrt(3*temp*const.KB/v_ave)

"""Position routines: distribute lattice, get box dimensions"""
def lattice_position(type = 'cubic'):
    """Distributes position over a lattice given the type

    Args:
        type -- (default = 'cubic') type of lattice: cubic or rectangular-z

    Errors:
        -- unspecified lattice type
    """
    if(type == 'cubic'):
        # generates position over a cubic lattice of a_lat = 1 (only positive positions)
        S_range = list(range(0,int(np.power(system.N, 1/system.dim))))
        cubic_lattice = np.array(list(itertools.product(S_range, repeat=system.dim)))

        # rescaling and shifting
        system.pos = cubic_lattice * (system.L[0]/np.cbrt(system.N)) + (system.L[0]/(2*np.cbrt(system.N)))

    elif(type == 'rectangular-z'):
        x = np.zeros(system.N)
        y = np.zeros(system.N)
        z = np.zeros(system.N)

        places_z = np.linspace(0, system.L[2], num=system.n[2], endpoint = False)
        places_z += places_z[1]*0.5
        n_part = 0

        for i,j,k in itertools.product(list(np.arange(system.n[0])), list(np.arange(system.n[1])), list(np.arange(system.n[2]))):

            x[n_part] = alat*i
            y[n_part] = alat*j
            z[n_part] = places_z[k]
            n_part += 1

        system.pos[:,0] = x
        system.pos[:,1] = y
        system.pos[:,2] = z

    else:
        print("Error: unknown lattice type.")

def get_box_dimensions():

    L = np.zeros(3)
    l = np.cbrt(0.5*system.N/system.rho)
    L[0] = l
    L[1] = l
    L[2] = l

    return L
