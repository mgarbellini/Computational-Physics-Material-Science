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
    file = open("6_results.txt", "w")
    K = np.loadtxt("./n6/LJMD_216_prod_energy.txt", usecols = 0)
    E = np.loadtxt("./n6/LJMD_216_prod_energy.txt", usecols = 2)
    U = np.loadtxt("./n6/LJMD_216_prod_energy.txt", usecols = 1)

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
    E = np.loadtxt("./n6/LJMD_216_equil_energy.txt", usecols = 2)
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
    while iter<216*3:
        xcols.append(iter)
        ycols.append(iter+1)
        zcols.append(iter+2)
        iter+=3

    velx = np.loadtxt("./n6/LJMD_216_prod_velocity.txt", usecols = xcols)

    sum_x = np.sum(velx*velx, axis = 1)

    mean_vel = np.mean(sum_x)
    var_vel = np.var(sum_x)
    std_vel = np.std(sum_x)

    file = open("6_velx_results.txt", "w")
    file.write("vel mean " + str(mean_vel)+ " var " + str(var_vel) + " std " + str(std_vel) + "\n")
    file.close()


    x = np.arange(len(sum_x))*10

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,sum_x, label="$\sum_{i}^N v_{i,x}^2$")
        ax.hlines(mean_vel, 0, len(x)*10 , color ="red", label = "Mean")
        #ax.autoscale(tight=True)
        ax.set_xlabel('timesteps')
        ax.set_ylabel('$\sum v_x^2 $ [reduced units]')
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

    x = [6,8,10]
    y = [52, 286, 1220]
    xf = np.linspace(0,10,100)
    yf = xf**(3.03)

    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,y, label ="Simulation data")
        ax.plot(xf,yf, label = "fit $N^{3.03\pm 0.09}$")

        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Seconds per simulation')
        ax.legend()
        ax.set_title("Implementation Efficiency")
        fig.savefig('./figures/efficiency.pdf')

if __name__ == '__main__':

    velocity_distribution()
