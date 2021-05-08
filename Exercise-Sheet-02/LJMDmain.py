#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 02 - Lennard-Jones fluid in microcanonical enesemble

*Objective*
Implement a molecular dynamics (MD) code that computes the trajectories of
a collection of N particles interacting through the Lennard-Jones (LJ)
potential in an (NVE) microcanonical ensemble.

Latest update: May 8th 2021
"""
# Core namespaces
import numpy as np
import sys

# Tailored modules/namespaces
import const # physical constants
import system # system descriptions and routine
import force # force routines
import settings # simulation input values and initialization
import printing # various printing and plotting routines


if __name__ == '__main__':
    settings.init()
    print(system.N)
