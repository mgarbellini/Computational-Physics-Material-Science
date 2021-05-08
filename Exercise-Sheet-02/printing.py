#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* PRINTING MODULE *
Contains all the routines for printing and saving computational
data to file. This includes basic plotting and ovito style outputs,
as well as printing routines for debugging

Latest update: May 8th 2021
"""

import sys
import time
import matplotlib.pyplot as plt

freq = None
energy_file = None
pos_file = None
vel_file = None
