#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* INTEGRATOR MODULE *
Contains integrations routines using known schemes:
- Velocity Verlet


Latest update: May 7th 2021
"""

import numpy as np
import system
import settings
import force
from numba import jit, njit, vectorize


# Routine for evolving the system and calculating trajectories. The algorithm implemented
# is the known Velocity-Verlet.
# //by default the routine computes a single iteration
#
# The Velocity-Verlet integrator is given by the following equations for position and velocities
# //note that in practice this equations are vectorial
# Eq(3) r(t+dt) = r(t) + v(t) * dt + 1/2m * dt^2 * f(t)
# Eq(4) v(t+dt) = v(t) + dt/2m * [f(t) + f(t+dt)]
#
# NOTE that a first force computation is needed outside this routine


def velocity_verlet(iterations = 1):
    iter = 0
    while iter < iterations:

        # update system positions
        # periodic boundary conditions need to be respected

        new_positions = system.pos + settings.DT*system.vel + 0.5*settings.DT**2*system.force/system.mass

        # boundary conditions for system bounded between walls on z-axis
        system.pos[:,0] = np.mod(new_positions[:,0], system.L[0])
        system.pos[:,1] = np.mod(new_positions[:,1], system.L[1])
        system.pos[:,2] = new_positions[:,2]

        if((new_positions[:,2] > system.L[2]).any()):
            print("UPPER ERROR: out of bound:")


        # save current force to local variable
        force_previous = system.force

        # force computation at new coordinates
        system.force, system.potential = force.lennard_jones(np.zeros(system.pos.shape, dtype = np.float), system.pos, system.L)
        system.force, system.potential, system.f_wall_dw, system.f_wall_up = force.lennard_jones_wall(system.pos, system.L, system.force, system.potential, system.f_wall_dw, system.f_wall_up)
        system.force = force.external_force(system.force,1)

        system.vel += 0.5*settings.DT*(system.force + force_previous)/system.mass

        # compute kinetic energy
        system.kinetic = 0.5*system.mass*np.sum(np.multiply(system.vel, system.vel))

        # update iter count when needed (by default iterations = 1)
        iter += 1
