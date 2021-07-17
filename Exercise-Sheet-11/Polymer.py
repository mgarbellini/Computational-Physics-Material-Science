#!/usr/bin/env python3

"""
@author:    M. Garbellini
@email:     matteo.garbellini@studenti.unimi.it

Albert-Ludwigs-Universität Freiburg
Computational Physics: Material Science
Exercise Sheet 10: Modelling a water nano-droplet


*Comments*:
- The bond contribution of the force seems to work perfectly, and indeed oscillates around the
average value r_not (See plot1 in current directory)

- The angle contribution however seems too be strong (about an order of magnitude) but I was not able
to find the origin of such problem

- Probably it is due to the fact that I was not able to properly define the initial hydrogen positions with
a specific angle_not value. Somehow the rotation matrix I was using gave inconsistent results.

Latest update: July 12th 2021
"""

# Core namespaces
import numpy as np
from tqdm import tqdm
from numba import jit, njit, vectorize
from numba import types, typed, typeof, deferred_type
import matplotlib.pyplot as plt

# Tailored modules/namespaces
from classmodule import Particle, Polymer_Chain, Force, IntegratorNH
from classmodule import KB, enot




if __name__ == '__main__':


    mass_particle = 2.56E-26
    integrator = IntegratorNH(0.5E-15, 300)
    force = Force(1058.,3.2E-10,3.2E-10,KB*integrator.T)



    """Initialize list of particles"""
    N = 51 #number of particles in polymer chain
    particles = typed.List()

    for i in range(N):
        particles.append(Particle(np.zeros(3), np.zeros(3), mass_particle, 0, 0))

    """Add particles to Integrator class"""
    integrator.add_particles(particles)

    """Initialize polymer chain"""
    polymer = Polymer_Chain(0,3.2E-10)
    polymer.load_particles(particles)
    polymer.distribute_particles()

    """Test force calculations"""
    force.add_polymer_chain(polymer)

    """End-to-End distance"""
    end_distance = []

    """Equilibration run"""
    for iter in tqdm(range(10000), desc = "Equilibration run"):

        if iter == 0: force.force_polymer_chain()
        if iter%500 == 0: end_distance.append(polymer.end_distance())
        integrator.half_step()
        force.initialize_force_polymer()
        force.force_polymer_chain()
        integrator.full_step()


    """Production run"""
    for iter in tqdm(range(100000), desc = "Production run"):

        integrator.half_step()
        force.initialize_force_polymer()
        force.force_polymer_chain()
        integrator.full_step()
        if iter%500 == 0: end_distance.append(polymer.end_distance())



    """Printing plot"""

    x = np.linspace(0, 110000*integrator.dt, num=len(end_distance))
    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot(x,np.asarray(end_distance))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("End-to-End distance [m]")
        ax.set_title("Mean squared distance")
        fig.savefig('./' + 'end_to_end' +'.pdf')













    """
    bond_len = []
    angle = []
    for iter in tqdm(range(10000)):

        if iter%100 == 0 : bond_len.append(water.get_bond_len())
        if iter%100 == 0 : angle.append(water.get_angle())
        integrator.half_step()
        force.initialize_forces()
        force.intra_molecular()
        integrator.full_step()

    average_bond = np.mean(np.asarray(bond_len))
    average_angle = np.mean(np.asarray(angle))
    print(average_bond, force.rnot)
    print(average_angle, force.anot)
    """

    #x = np.linspace(0, 10000*integrator.dt, num=len(bond_len))
    #printing.plot(False, x, bond_len, "Bond Length", "Time [s]", "Bond Length [m]", "Bond Length", "bond_len")
