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
import printing
import routines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import settings

def analysis(file):

    filename = "data_0" + file + ".txt"

    T = np.loadtxt(filename, usecols = 0)
    P = np.loadtxt(filename, usecols = 1)
    CVE = np.loadtxt(filename, usecols = 2)
    CVU = np.loadtxt(filename, usecols = 3)
    KT = np.loadtxt(filename, usecols = 4)

    """Pressure and temperature plots
    1. reduce sampling frequency
    2. plot T(t) and P(t)
    """

    temp = []
    pres = []
    for i in range(len(T)):
        if i%10 == 0:
            temp.append(T[i])
            pres.append(P[i])

    x = np.linspace(0, 45000, num = len(temp))
    printing.plot(False, x, temp, "$T(t)$", "Iterations", "Temperature [K]", "Temperature over time", "T_" + filename)
    printing.plot(False, x, pres, "$P(t)$", "Iterations", "Pressure [Pa]", "Pressure over time", "P_" + filename)

    """Block averages and errors
    1. divide iterations into blocks
    2. average over block
    3. compute statistical errors
    """

    block_P = []
    block_CVE = []
    block_CVU = []
    block_KT = []

    for i in range(9):
        block_P.append(np.mean(P[(100*i):(100*(i+1)-1)]))
        block_CVE.append(np.mean(CVE[(100*i):(100*(i+1)-1)]))
        block_CVU.append(np.mean(CVU[(100*i):(100*(i+1)-1)]))
        block_KT.append(np.mean(KT[(100*i):(100*(i+1)-1)]))

    error_P = np.std(np.asarray(block_P)) / np.sqrt(len(block_P))
    error_CVE = np.std(np.asarray(block_CVE)) / np.sqrt(len(block_P))
    error_CVU = np.std(np.asarray(block_CVU)) / np.sqrt(len(block_P))
    error_KT = np.std(np.asarray(block_KT)) / np.sqrt(len(block_P))

    plot_error_bars(block_P, error_P,"Pressure [Pa]", "Pressure over time", "Pressure",  file)
    plot_error_bars(block_CVE, error_CVE,"$Cv$ [$J$/$Kg K$]", "Specific heat with $\\sigma_E^2$ ", "SpecifiHeatE",file)
    plot_error_bars(block_CVU, error_CVU,"$Cv$ [$J$/$Kg K$]", "Specific heat with $\\sigma_U^2$", "SpecifiHeatU",file)
    plot_error_bars(block_KT, error_KT,"$k_T$ [$m^2$/$N$]", "Isothermal Compressibility", "Compressibility",file)

    av_pressure = np.mean(P)
    print(file, av_pressure)


def plot_error_bars(y,yerror,ylabel, title, filename1, filename2):
    with plt.style.context(['science']):

        x = np.linspace(0, 45000, num = len(y))
        fig, ax = plt.subplots()
        ax.errorbar(x,y,yerr=yerror, marker = 'o', ms = 4, color = 'green', label = "Block Average")

        ax.set_xlabel("Iterations")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.savefig('./' + filename1 + '_' +filename2 + '.pdf')


def block_average(block_size):
    P = np.loadtxt("data_05.txt", usecols = 1)
    block_P = []

    tot = 900 #number of points
    n_blocks = int(900/block_size)
    for i in range(n_blocks):
        block_P.append(np.mean(P[(block_size*i):(block_size*(i+1))]))
        i += 1
    error_P = np.std(np.asarray(block_P)) / np.sqrt(len(block_P))

    return error_P









if __name__ == '__main__':

    block_size = np.arange(5000)+1
    error = []
    P = np.loadtxt("data_05.txt", usecols = 0)
    r_average = routines.running_average(P,1)
    for i in range(len(block_size)):
        ave, err = routines.block_average(P, block_size[i])
        error.append(err)

    printing.plot(False, block_size, error, "$\\frac{\\sigma_b}{\\sqrt{N_b}}$", "Block size", "Statistical Error", "Error Convergence for T", "stat_error")
    x = np.linspace(0, 300000, num=len(P))
    printing.plot(False, x, r_average, "$\\frac{\\sigma_b}{\\sqrt{N_b}}$", "Block size", "Statistical Error", "Running Average", "running")


    """
    #filenames = ["5", "1", "05", "01", "005"]
    filenames = ["005",]
    for filename in filenames:
        analysis(filename)
    """

    """
    P = [1244290,2483871,12025790, 23454799, 155413035]
    rho = [0.005, 0.01, 0.05, 0.1, 0.5]
    sigma = 2.55E-10
    kb = 1.38E-23
    T = 300

    diff = []

    for i in range(len(P)):
        diff.append(P[i]-T*kb*rho[i]/(sigma**3))

    with plt.style.context(['science']):

        fig, ax = plt.subplots()
        ax.plot(rho,diff,marker = 'o', ms = 4, color = 'green', label = r"$P-\rho k_B T$")

        ax.set_xlabel(r"$\rho[m^-3]$")
        ax.set_ylabel(r"$P-P_{id}$")
        ax.set_title("Pressure Deviation")
        ax.legend()
        fig.savefig('./pressure_deviation.pdf')
    """
