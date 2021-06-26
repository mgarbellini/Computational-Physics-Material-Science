#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

* STRUCTURE FACTOR ANALYSIS (EXERCISE SHEET 07) *
Contains the analysis routines for computing the structure
factor as explained in Exercise Sheet 07, given a file "positions.txt"
as input. Thus this is a standalone script

Latest update: June 21st 2021
"""
import numpy as np
import routines
import settings
import system
import force
import const
import printing
from tqdm import tqdm
from numba import jit, njit, vectorize



def import_positions():

    pos = np.loadtxt("positions.txt")
    n_particles = int(pos.shape[1]/3)

    colx = []
    coly = []
    colz = []

    i = 0
    part = 0
    while i < n_particles*3:
        colx.append(i)
        coly.append(i+1)
        colz.append(i+2)
        part+=1
        i=i+3

    posx = np.transpose(np.loadtxt("positions.txt", usecols = colx))
    posy = np.transpose(np.loadtxt("positions.txt", usecols = coly))
    posz = np.transpose(np.loadtxt("positions.txt", usecols = colz))

    pos = np.stack((posx, posy, posz), axis=1)
    return posx, posy, posz, pos

def wave_vector(max=50):

    return np.arange(max)*2*np.pi/system.L[0]

@njit
def structure_factor(posx, posy, posz, k):

    S = 0
    for iter in range(posx.shape[1]):
        sumcos = 0
        sumsin = 0
        for i in range(posx.shape[0]):
            sumcos += np.cos(k*(posx[i, iter] + posy[i, iter] + posz[i, iter]))
            sumsin += np.sin(k*(posx[i, iter] + posy[i, iter] + posz[i, iter]))

        S += (sumcos*sumcos + sumsin*sumsin)/posx.shape[0]

    return S/posx.shape[1]

@njit
def structure_factor_integral(g,x,k,rho):
    """Integration performed with trapezoidal method"""
    S = 1
    const = 4*np.pi*rho
    for i in range(g.shape[0]):
        S += const*((g[i]-1)*x[0]*x[i]*x[i]*np.sin(k*x[i])/k/x[i])

    return S

if __name__ == '__main__':

    settings.init()
    posx, posy, posz, pos = import_positions()
    k = wave_vector()
    knorm = np.sqrt(3)*k



    """Check that positions are correct by computing radial distribution function"""
    rdf = 0
    isothermal = []
    for i in tqdm(range(pos.shape[2]), desc = "Radial distribution function"):
        system.pos = pos[:,:,i]
        radial, bins, kt = routines.radial_distribution_function()
        isothermal.append(kt)
        rdf += radial

    rdf = rdf/pos.shape[2]
    xx = np.linspace(0,pos.shape[2], num=len(isothermal))
    printing.plot(False, bins, rdf, "$g(r)$", "$r$ [$\\sigma$]", "$g(r)$", "RDF with $\\rho = 0.5\\sigma^{-3}$", "Radial")
    printing.plot(False, xx, isothermal, "$k_T(t)$", "Iterations", "$k_T$", "Isothermal Compressibility", "Isothermal")


    """Computes the structure factor for all iterations"""
    S = []
    S_int = []
    for i in tqdm(range(len(k)), desc = "Structure factor"):
        S.append(structure_factor(posx, posy, posz, k[i]))
        if i>0:
            S_int.append(structure_factor_integral(rdf, bins*force.sigma, knorm[i], system.rho))

    printing.plot(True, knorm[1:], [(np.asarray(S))[1:], np.asarray(S_int)], ["Sum of sin and cos", "Integral of $g(r)$"], "$|k|$", "$S(k)$", "Structure factor", "structure_factor")

    """isothermal compressibility"""
    S_to_zero = structure_factor_integral(rdf,bins*force.sigma, 0.5, system.rho)
    kt = S_int[0]/system.rho/const.KB/system.T
    kt0 = S_to_zero/system.rho/const.KB/system.T


    """isothermal compressibility near zero """
    k = np.linspace(0.01, 2, num=20)*np.sqrt(3)*2*np.pi/system.L[0]
    kplot = np.linspace(0.01, 2, num=20)
    ktint = []
    ktgr = []
    kt = []
    structure = 0
    for i in tqdm(range(len(k)), desc = "Isothermal compressibility"):
        structure = structure_factor_integral(rdf,bins*force.sigma, k[i], system.rho)
        ktgr.append(structure_factor(posx, posy, posz, k[i]))
        ktint.append(structure/system.rho/const.KB/system.T)
        kt.append(isothermal[-1])

    printing.plot(True, kplot, [np.asarray(ktint), np.asarray(kt)], ["with $\\lim_{k \\to 0} S(k)$", "Explicit $k_T$ equation"], "$|k|$", "$k_T$", "Isothermal compressibility", "isothermal_compressibility_compared")
