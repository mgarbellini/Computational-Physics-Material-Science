import numpy as np
import sys
import itertools
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import integrator

N = 2

# generates position over a cubic lattice of a_lat = 1 (only positive positions)
S_range = list(range(0,N))
cubic_lattice = np.array(list(itertools.product(S_range, repeat=3)))
cubic_lattice = cubic_lattice * (1/N)



# rescaling and shifting

#compute shortest distance
# r_ij
X = np.transpose(cubic_lattice[:,0] * np.ones((len(cubic_lattice[:,0]), len(cubic_lattice[:,0]))))
Y = np.transpose(cubic_lattice[:,1] * np.ones((len(cubic_lattice[:,1]), len(cubic_lattice[:,1]))))
Z = np.transpose(cubic_lattice[:,2] * np.ones((len(cubic_lattice[:,2]), len(cubic_lattice[:,2]))))
r_x = X - np.transpose(X)
r_y = Y - np.transpose(Y)
r_z = Z - np.transpose(Z)
r = np.sqrt(r_x**2 + r_y**2 + r_z**2)+ np.eye(N**3)
r_rec = np.reciprocal(r) - np.eye(N**3)
f_x = r_rec**13 - r_rec**7
