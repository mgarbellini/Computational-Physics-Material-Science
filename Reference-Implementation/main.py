# MD code in microcanonical ensemble
# force field = LJ with eps = alpha * kb * T and r0 = 15 angstrom
# the treatment of the discontinuity at the cutoff is not implemented, i.e. the potential does not go smoothly to 0 at r = cutoff

import settings
import initialize
import force
import update
import debug
import sys

import misc


fileoutput = open("output_equilibration.txt", "w")
fileenergy = open("energy_equilibration.txt", "w")
fileenergy.write("step  PE  KE \n")

# initialization of global variable
settings.init()

# create atomic locations and velocities + cancel linear momentum + rescale velocity to desired temperature
initialize.InitializeAtoms()

# initialize the forces
force.forceLJvec()


if settings.debug == 1:
    debug.WriteForces(0)
ngr = 0
for step in range(0, settings.nsteps):

    if settings.debug == 1:
        print("step == ",step)

    update.VelocityVerletVec()

    update.KineticEneryVec()

    if step % 10 == 0:  #save trajectory and energies every 10 steps
        misc.WriteEnergy(fileenergy, step)
    if step % 500 == 0:  #save trajectory and energies every 10 steps
        misc.WriteTrajectory(fileoutput, step)

    if settings.Trescale  == 1 and step % 20 == 0: # rescaling of the temperature # the following lines should be defined as a routine in misc
            Trandom = initialize.temperature()
            initialize.rescalevelocity(settings.Tdesired, Trandom)

            Trandom1 = initialize.temperature()
            #print("after rescaling of the velocities at step ",step," Trandom == ",Trandom, Trandom1)


print("we finished the equilibration")


fileoutput.close()
fileenergy.close()

# we start colling down

fileoutput = open("output_production.txt","w")
fileenergy = open("energy_production.txt","w")
ngr = 0
for step in range(settings.nsteps, 2*settings.nsteps):

    update.VelocityVerletVec()

    update.KineticEneryVec()

    if step % 10 == 0:  #save trajectory and energies every 10 steps
        misc.WriteEnergy(fileenergy, step)
    if step % 100 == 0:  #save trajectory and energies every 10 steps
        misc.WriteTrajectory(fileoutput, step)


    if settings.Trescale  == 1 and step % 20 == 0: # rescaling of the temperature # the following lines should be defined as a routine in misc
            Trandom = initialize.temperature()

            initialize.rescalevelocity(settings.Tdesired, Trandom)

            Trandom1 = initialize.temperature()




print("we finished the production run")


fileoutput.close()
fileenergy.close()
