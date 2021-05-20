#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* ANALYSIS *
Contains analysis routines
-
-
-

Latest update: May 10th 206
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import settings
import scipy.stats

def energy_production():
    file = open("4_results.txt", "w")
    K = np.loadtxt("./LJMD_64_prod_energy.txt", usecols = 0)
    E = np.loadtxt("./LJMD_64_prod_energy.txt", usecols = 2)
    U = np.loadtxt("./LJMD_64_prod_energy.txt", usecols = 1)

    mean_K = np.mean(K)
    var_K = np.var(K)
    std_K = np.std(K)

    mean_U = np.mean(U)
    var_U = np.var(U)
    std_U = np.std(U)

    mean_E = np.mean(E)
    var_E = np.var(E)
    std_E = np.std(E)

    file.write("E mean " + str(mean_E)+ " var " + str(var_E) + " std " + str(std_E) + "\n")
    file.write("K mean " + str(mean_K)+ " var " + str(var_K) + " std " + str(std_K) + "\n")
    file.write("U mean " + str(mean_U)+ " var " + str(var_U) + " std " + str(std_U) + "\n")
    file.close()
    x = np.arange(len(K))
    x = x*10


    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,K, label="Kinetic Energy")
        ax.hlines(mean_K, 0, len(x)*10 , color ="red", label = "Mean")
        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('K [reduced units]')
        ax.legend()
        ax.set_title("Kinetic energy over the number of steps")
        fig.savefig('./figures/6_kinetic.pdf')

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,U, label="Potential Energy")
        ax.hlines(mean_U, 0, len(x)*10 , color ="red", label = "Mean")
        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('U [reduced units]')
        ax.legend()
        ax.set_title("Potential energy over the number of steps")
        fig.savefig('./figures/6_potential.pdf')

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,E, label="Total Energy")
        ax.hlines(mean_E, 0, len(x)*10 , color ="red", label = "Mean")
        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('E [reduced units]')
        ax.legend()
        ax.set_title("Total energy over the number of steps")
        fig.savefig('./figures/6_totalenergy.pdf')

def energy_equilibration():
    E = np.loadtxt("./LJMD_64_equil_energy.txt", usecols = 2)
    mean_E = np.mean(E)
    x = np.arange(len(E))*10

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,E, label="Total Energy")
        ax.hlines(mean_E, 0, len(x)*10 , color ="red", label = "Mean Energy")
        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('E [reduced units]')
        ax.legend()
        ax.set_title("Total energy during equilibration")
        fig.savefig('./figures/6_equil_totalenergy.pdf')

def velocity():

    xcols = []
    ycols = []
    zcols = []

    iter = 0
    while iter<64*3:
        xcols.append(iter)
        ycols.append(iter+1)
        zcols.append(iter+2)
        iter+=3

    velx = np.loadtxt("./LJMD_64_prod_velocity.txt", usecols = xcols)
    vely = np.loadtxt("./LJMD_64_prod_velocity.txt", usecols = ycols)
    velz = np.loadtxt("./LJMD_64_prod_velocity.txt", usecols = zcols)

    sum_x = np.sum(velx*velx, axis = 1)
    sum_y = np.sum(vely*vely, axis = 1)
    sum_z = np.sum(velz*velz, axis = 1)

    x = np.arange(len(sum_x))*10

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,sum_x, label="$\sum_{i}^N v_{i,x}^2$")
        #ax.plot(x,sum_y, label="$\sum_{i}^N v_{i,y}^2$")
        #ax.plot(x,sum_z, label="$\sum_{i}^N v_{i,z}^2$")

        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('$\sum v_a^2 $ [reduced units]')
        ax.legend()
        ax.set_title("Sum of squared velocities")
        fig.savefig('./figures/6_prod_velocity_x.pdf')

def velocity_distribution():

    xcols = []
    ycols = []
    zcols = []

    iter = 0
    while iter<1000*3:
        xcols.append(iter)
        ycols.append(iter+1)
        zcols.append(iter+2)
        iter+=3

    velx = np.loadtxt("./n10/LJMD_1000_prod_velocity.txt", usecols = xcols)
    vely = np.loadtxt("./n10/LJMD_1000_prod_velocity.txt", usecols = ycols)
    velz = np.loadtxt("./n10/LJMD_1000_prod_velocity.txt", usecols = zcols)

    velx = velx[-1,:]*velx[-1,:]
    velz = velz[-1,:]*velz[-1,:]
    vely = vely[-1,:]*vely[-1,:]

    velsum = velx + vely + velz
    min = np.min(velsum)
    max = np.max(velsum)
    K = np.mean(velsum)*2

    x = np.linspace(0, 20, 10000)
    maxwell_dist = np.exp(-x**2/(2*1.5*K))*x/(np.sqrt(1.5*K*2*np.pi))/2
    maxx = np.exp(-x**2/4)/(np.sqrt(4*np.pi))

    data = scipy.stats.maxwell.rvs(size=100000, loc=0, scale=np.sqrt(1.5*K))


    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        #ax.hist(velx, bins = 50, density = True, histtype='step')
        #ax.hist(vely, bins = 50, density = True, histtype='step')
        #ax.hist(velz, bins = 50, density = True, histtype='step')
        ax.hist(velsum, bins = 50, density = False , histtype='step')



        ax.set_xlabel('$v^2$ [reduced units]')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.set_title("Distribution $P(v^2)$")
        fig.savefig('./velocity_distribution.pdf')

def efficiency():

    x = [2**3,3**3,4**3,5**3,6**3,7**3,8**3,9**3,10**3]
    y = [0.85,1.5,5,16,44,113,265,577,1220]
    z = [5,5,9,20,49,116,259,530,1032]

    w = [0.88,1,1.35,2.7,6,14,31,60,116]
    xf = np.linspace(0,1000,100000)
    yf = xf**(0.99)
    zf = xf**(0.97)
    wf = xf**(0.63)

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,y, label ="Numpy")
        ax.plot(x,z, label ="Numpy+Numba")
        ax.plot(x,w, label ="Numba")


        #ax.plot(xf,yf)
        #ax.plot(xf,zf)
        #ax.plot(xf,wf)

        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Seconds per 4000 iterations')
        ax.legend()
        ax.set_title("Code performance: Numpy vs Numba")
        fig.savefig('./efficiency.pdf')

if __name__ == '__main__':

    velocity()
    energy_production()
