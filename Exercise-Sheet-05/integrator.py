#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* INTEGRATOR MODULE *
Contains integrations routines using known schemes:
- Velocity Verlet
- Half step Velocity Verlet in NH Thermostat


Latest update: June 6th 2021
"""

import numpy as np
import system
import settings
import time
import force
import routines
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

    return k*mass/2


"""GENERAL VELOCITY-VERLET INTEGRATION SCHEME"""


@njit
def vv_pos(pos, dt, vel, force, mass, L):
    """Computes the new positions using the velocity-verlet integration scheme

    Args:
        pos -- np.array(N,dim) containing the current particle positions
        dt -- timestep
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
            pos[i, dim] = np.mod(pos[i,dim]+ dt*vel[i,dim] + 0.5*dt**2*force[i,dim]/mass, L[dim])

    return pos


@njit
def vv_vel(vel, dt, force, force_previous, mass):
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
            vel[i, dim] += 0.5*dt * (force[i, dim] + force_previous[i, dim])/mass

    return vel


def velocity_verlet():

    """Updates the positions and velocities according to the velocity-verlet
    integration scheme

    Note:
        -- the routine outsources the computations to specific Numba functions
    """

    """Update positions"""
    #system.pos = vv_pos(system.pos, settings.DT, system.vel, system.force, system.mass, system.L)
    vtime = time.perf_counter()

    new_pos = system.pos+ settings.DT*system.vel + 0.5*settings.DT*settings.DT*system.force/system.mass
    system.pos[:,0] = np.mod(new_pos[:,0], system.L[0])
    system.pos[:,1] = np.mod(new_pos[:,1], system.L[1])
    system.pos[:,2] = np.mod(new_pos[:,2], system.L[2])



    """Save current force to local variable for later calculation"""
    force_previous = system.force

    """Force computation at new coordinates"""
    ftime = time.perf_counter()
    system.force, system.potential = force.lennard_jones(np.zeros(system.force.shape, dtype=np.float), system.pos, system.L)
    print("Force Iteration time: ", time.perf_counter()-ftime)

    """Update velocities"""
    #system.vel = vv_vel(system.vel, settings.DT, system.force, force_previous, system.mass)
    system.vel += 0.5*settings.DT * np.divide((system.force + force_previous),system.mass)

    """Compute kinetic energy"""
    #system.kinetic = compute_kinetic(system.vel, system.mass)
    system.kinetic = 0.5*system.mass*np.sum(np.multiply(system.vel, system.vel))

    print("Velocity-Verlet time: ", time.perf_counter()-vtime)
    system.energy = system.kinetic + system.potential





"""NOSE-HOOVER INTEGRATION - HALF STEP VELOCITY VERLET"""


@njit
def nh_vel1(v, f, xi, m, dt):
    """Computes the half step velocity using the nose-hoover velocity verlet

    Args:
        v -- np.array(N,dim) containing current particles velocities
        f -- np.array(N,dim) containing current particles net forces
        xi -- current value of xi
        m -- particle mass
        dt -- timestep

    Returns:
        v -- np.array(N,dim) containing updated velocities (half step)

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for dim in range(v.shape[1]):
        for i in range(v.shape[0]):
            v[i, dim] = (v[i, dim] + 0.5*f[i, dim]*dt/m)/(1+xi*dt/2)
    return v

@njit
def nh_pos(v, p, dt):
    """Updates position using the Nose-Hoover half step velocity verlet

    Args:
        v -- np.array(N,dim) containing current particles velocities
        p -- np.array(N,dim) containing current particles positions
        dt -- timestep

    Returns:
        p -- np.array(N,dim) containing updated positions

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for dim in range(p.shape[1]):
        for i in range(p.shape[0]):
            p[i, dim] += v[i, dim]*dt

    return p

@njit
def nh_vel2(v, f, m, xi, dt):
    """Computes the full step velocity using the nose-hoover velocity verlet

    Args:
        v -- np.array(N,dim) containing current particles velocities
        f -- np.array(N,dim) containing current particles net forces
        xi -- current value of xi
        m -- particle mass
        dt -- timestep

    Returns:
        v -- np.array(N,dim) containing updated velocities (full step)

    Notes:
        -- the routine runs using Numba @njit decorator for faster run time
    """
    for dim in range(v.shape[1]):
        for i in range(v.shape[0]):
            v[i, dim] = v[i, dim] + 0.5*dt*(f[i, dim]/m - xi*v[i, dim])

    return v

@njit
def nh_xi(xi, G_half, dt):

    return xi + G_half*dt

@njit
def nh_lns(lns, xi, G_half, dt):

    return lns + xi*dt + G_half*dt*dt/2

"""
def nh_integrate(v, ):

    # vel half
    for dim in range(v.shape[1]):
        for i in range(v.shape[0]):
            v[i, dim] = (v[i, dim] + 0.5*f[i, dim]*dt/m)/(1+xi*dt/2)

    # kinetic_half
    k = 0
    for dim in range(v.shape[1]):
        for i in range(v.shape[0]):
            k += v[i, dim]*v[i, dim]

    k = k*mass/2

    # G half
    G_half = (2*k - 3*N*kb*T)/Q

    # lns
    lns += xi*dt + G_half*dt*dt/2

    # xi
    xi += G_half*dt

    # force
    force, potential = force.lennard_jones(
        np.zeros(system.pos.shape, dtype=np.float), system.pos, system.L)
"""


def nose_hoover_integrate(iter):
    """Updates the Nose-Hoover Thermostat using an half-step velocity verlet

    """

    vel_half = nh_vel1(system.vel, system.force,
                       system.xi, system.mass, settings.DT)

    system.pos = nh_pos(vel_half, system.pos, settings.DT)

    kinetic_half = compute_kinetic(vel_half, system.mass)

    G_half = routines.compute_G(kinetic_half,system.N, settings.kb, system.T, system.Q)

    system.lns = system.lns + system.xi*settings.DT + G_half*settings.DT*settings.DT/2

    system.xi = system.xi + G_half*settings.DT

    system.force, system.potential = force.lennard_jones(
        np.zeros(system.pos.shape, dtype=np.float), system.pos, system.L)

    system.vel = nh_vel2(vel_half, system.force,
                         system.mass, system.xi, settings.DT)

    """Compute Energies"""
    if iter%settings.sampling_freq == 0:
        system.kinetic = compute_kinetic(system.vel, system.mass)
        system.nose_hoover = routines.nose_hoover_energy(system.Q, system.xi, system.N, settings.kb, system.T, system.lns)
        system.energy = system.kinetic + system.potential + system.nose_hoover
