#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* INTEGRATOR MODULE *
Contains integrations routines using known schemes:
- Verlet
- Velocity Verlet
- Euler

Latest update: May 7th 2021
"""

import numpy as np
import system
import settings
import force


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
def VelocityVerlet(iterations = 1):
    iter = 0
    while iter < iterations:

        # update system positions
        # periodic boundary conditions need to be respected
        new_positions = (system.pos + settings.DT*system.vel + 0.5*settings.DT**2*system.force/system.mass)
        system.pos = np.fmod(new_positions, system.L)

        # save current force to local variable
        force_previous = system.force

        # force computation at new coordinates
        force.LennardJones()
        system.vel += 0.5*settings.DT*(system.force + force_previous)/system.mass

        # update iter count when needed (by default iterations = 1)
        iter += 1
