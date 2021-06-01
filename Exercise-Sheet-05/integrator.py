#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* INTEGRATOR MODULE *
Contains integrations routines using known schemes:
- Velocity Verlet
- Half step Velocity Verlet in NH Thermostat


Latest update: June 1st 2021
"""

import numpy as np
import system
import settings
import force
from numba import jit, njit, vectorize

"""GENERAL VELOCITY RELATED ROUTINES"""
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

"""GENERAL VELOCITY-VERLET INTEGRATION SCHEME"""
@njit
def vv_pos(pos, DT, vel, force, mass, L):
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
            pos[i,dim] += + DT*vel[i,dim] + 0.5*DT**2*force[i,dim]/mass
            pos[i,dim] = np.mod(pos[i,dim], L[dim])

    return pos

@njit
def vv_vel(vel, DT, force, force_previous, mass):
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
            vel[i,dim] += vel[i,dim] + 0.5*DT*(force[i,dim] + force_previous[i,dim])/mass

    return vel


def velocity_verlet():
    """Updates the positions and velocities according to the velocity-verlet
    integration scheme

    Note:
        -- the routine outsources the computations to specific Numba functions
    """

    """Update positions"""
    system.pos = vv_pos(system.pos, settings.DT, system.vel, system.force, system.mass, system.L)

    """Save current force to local variable for later calculation"""
    force_previous = system.force

    """Force computation at new coordinates"""
    system.force, system.potential = force.lennard_jones(np.zeros(system.pos.shape, dtype = np.float), system.pos, system.L)

    """Update velocities"""
    system.vel = vv_vel(system.vel, settings.DT, system.force, force_previous, system.mass)

    """Compute kinetic energy"""
    system.kinetic = compute_kinetic(system.vel, system.mass)


"""NOSE-HOOVER INTEGRATION - HALF STEP VELOCITY VERLET"""

@njit
def nh_vel1():

@njit
def nh_pos():

@njit
def nh_lns():

@njit
def nh_xi():

@njit
def nh_vel2():


def nose_hoover_integrate():
    """Updates the Nose-Hoover Thermostat using an half-step velocity verlet

    """

    vel_half = nh_vel1(system.vel, system.force, system.xi, system.mass, settings.DT)

    system.pos = nh_pos(vel_half, system.pos, settings.DT)

    kinetic_half = compute_kinetic(vel_half, system.mass)

    G_half = routines.compute_G(kinetic_half)

    system.lns = nh_lns(system.lns, system.xi, G_half, settings.DT)

    system.xi = nh_xi(system.xi, G_half, settings.DT)

    system.force, system.potential = force.lennard_jones(np.zeros(system.pos.shape, dtype = np.float), system.pos, system.L)

    system.vel = nh_vel2(vel_half, system.force, system.mass, system.xi, settings.DT)

    system.kinetic = compute_kinetic(system.vel, system.mass)
