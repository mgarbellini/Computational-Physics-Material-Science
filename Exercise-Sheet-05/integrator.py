#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* INTEGRATOR MODULE *
Contains integrations routines using known schemes:
- Velocity Verlet


Latest update: May 26th 2021
"""

import numpy as np
import system
import settings
import force
from numba import jit, njit, vectorize



@njit
def position_update(pos, DT, vel, force, mass, L):
    """Computes the new positions using the velocity-verlet integration scheme

    Args:
        pos -- np.array(N,dim) containing the current particle positions
        DT -- timestep
        vel -- np.array(N,dim) containing the current particle velocities
        force -- np.array(N,dim) containing the current net forces on each particle
        mass -- value of particle mass

    Returns:
        pos -- np.array(N,dim) containing updated positions

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for dim in range(pos.shape[1]):
        for i in range(pos.shape[0]):
            pos[i,dim] += DT*vel[i,dim] + 0.5*DT**2*force[i,dim]/mass
            pos[i,dim] = np.mod(pos[i,dim], L[dim])

    return pos

@njit
def velocity_update(vel, DT, force, force_previous, mass):
    """Computes the new velocities using the velocity-verlet integration scheme

    Args:
        vel -- np.array(N,dim) containing the current particle velocities
        DT -- timestep
        force -- np.array(N,dim) containing the current net forces on each particle
        force_previous -- np.array(N,dim) contaning forces at previous timestep
        mass -- value of particle mass

    Returns:
        vel -- np.array(N,dim) containing updated velocities

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for dim in range(vel.shape[1]):
        for i in range(vel.shape[0]):
            vel[i,dim] += 0.5*DT*(force + force_previous)/mass

    return vel

@njit
def compute_kinetic(vel, mass):
    """Computes the kinetic energy of the system

    Args:
        vel -- np.array(N,dim) containing all the particle velocities
        mass -- value of particle mass

    Returns:
        -- kinetic energy of the system

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    k = 0
    for dim in range(vel.shape[1]):
        for i in range(vel.shape[0]):
            k += vel[i, dim]*vel[i, dim]

    return k/mass/2

def velocity_verlet():
    """Updates the positions and velocities according to the velocity-verlet
    integration scheme

    Note:
        -- the routine outsources the computations to specific Numba functions
    """

    """Update positions"""
    system.pos = position_update(system.pos, settings.DT, system.vel, system.force, system.mass, system.L)

    """Save current force to local variable for later calculation"""
    force_previous = system.force

    """Force computation at new coordinates"""
    system.force, system.potential = force.lennard_jones(np.zeros(system.pos.shape, dtype = np.float), system.pos, system.L)

    """Update velocities"""
    system.vel = velocity_update(system.vel, settings.DT, system.force, force_previous, system.mass)

    """Compute kinetic energy"""
    system.kinetic = compute_kinetic(system.vel, system.smass)
