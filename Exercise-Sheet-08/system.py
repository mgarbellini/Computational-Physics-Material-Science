#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* SYSTEM MODULE *
Contains the system variables

Latest update: June 2nd 2021
"""

import numpy as np

"""System variables"""
ensemble = None  # type of ensemble ("microcanonical, NVT, NHT")
dim = None  # dimension of the system (2D or 3D)
alat = None  # Lattice parameter
rho = None  # Number density
p = None
L = None  # Box dimensions (per edge)

"""Particles variables"""
mass = None
n = None
N = None
pos = None
vel = None

"""Elecrostatic variables"""
charge = None
e = 1.60217663E-19
surface_charge = None
discrete_surface_q_pos = None
discrete_surface_q = None


"""Force variables"""
force = None
f_wall_dw = None  # force on lower wall
f_wall_up = None  # force on upper wall
external_force = False

"""Energy and thermodynamics variables"""
energy = None
kinetic = None
potential = None
T = None
kt = None  # Isothermal compressibility
cv = None  # Heat capacity

"""Nose-Hoover Thermostat specific variables"""
Q = None  # Termal mass
lns = None  # Lagrangian fictitous degree of freedom (log of s)
xi = None
G = None
nose_hoover = None  # Energy contribution of the NH thermostat

"""Thermostat/Barostat variables"""
virial = None #Internal virial
pressure = None #Pressure of the system
