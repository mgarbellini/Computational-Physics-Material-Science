#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* CONSTANTS MODULE *
Contains all the major physical constants needed for simulating
a system. Units of measure are also provided

Latest update: May 8th 2021
"""

import numpy as np
import settings
import system
import force

KB = 1
# This routine converts real-units to the appropriate reduced-units
# for a Lennard-Jones liquid simulation
# NOTE: implementation yet to be finished
def LennardJonesReducedUnits():

    # Temperature conversion
    # T* = T Kb/epsilon (epsilon is in units of KbT)
    system.T = 1/force.epsilon

    # Timestep conversion
    # DT* = DT sigma sqrt(mass/epsilon)
    settings.DT = settings.DT
