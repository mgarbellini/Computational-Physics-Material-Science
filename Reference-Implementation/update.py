import settings
import force
import numpy as np



def VelocityVerletVec():




    # update the position at t+dt
    settings.xi += settings.vix * settings.deltat + settings.sumfix * settings.deltat * settings.deltat *0.5 / settings.mass
    settings.yi += settings.viy * settings.deltat + settings.sumfiy * settings.deltat * settings.deltat *0.5 / settings.mass
    settings.zi += settings.viz * settings.deltat + settings.sumfiz * settings.deltat * settings.deltat *0.5 / settings.mass

    # save the force at t
    sumfixati = settings.sumfix
    sumfiyati = settings.sumfiy
    sumfizati = settings.sumfiz

    # update acceleration at t+dt
    force.forceLJvec()

    # update the velocity
    settings.vix += 0.5*settings.deltat *(settings.sumfix + sumfixati) / settings.mass
    settings.viy += 0.5*settings.deltat *(settings.sumfiy + sumfiyati) / settings.mass
    settings.viz += 0.5*settings.deltat *(settings.sumfiz + sumfizati) / settings.mass



def VelocityVerlet():

    fx0 = []
    fy0 = []
    fz0 = []
    dt = settings.deltat
    mass = settings.mass
    #update the position at t+dt
    i = 0
    while i < settings.nparticles:
        settings.p[i].x += settings.p[i].vx * dt + settings.p[i].fx * dt * dt * 0.5 / mass
        settings.p[i].y += settings.p[i].vy * dt + settings.p[i].fy * dt * dt * 0.5 / mass
        settings.p[i].z += settings.p[i].vz * dt + settings.p[i].fz * dt * dt * 0.5 / mass
        i += 1

    # save the force at t
    i = 0
    while i < settings.nparticles:
        fx0.append(settings.p[i].fx)
        fy0.append(settings.p[i].fy)
        fz0.append(settings.p[i].fz)
        i += 1

    # update acceleration at t+dt
    force.forceLJ()

    # update the velocity
    i = 0
    while i < settings.nparticles:

        settings.p[i].vx += 0.5 * dt * (settings.p[i].fx + fx0[i]) / mass
        settings.p[i].vy += 0.5 * dt * (settings.p[i].fy + fy0[i]) / mass
        settings.p[i].vz += 0.5 * dt * (settings.p[i].fz + fz0[i]) / mass
        i += 1

def KineticEneryVec():


    mass = settings.mass

    settings.Energy.ekx = 0.5*mass * np.sum(np.multiply(settings.vix, settings.vix) )
    settings.Energy.eky = 0.5*mass * np.sum(np.multiply(settings.viy, settings.viy) )
    settings.Energy.ekz = 0.5*mass * np.sum(np.multiply(settings.viz, settings.viz) )

    settings.Energy.ek = settings.Energy.ekx + settings.Energy.eky + settings.Energy.ekz


def KineticEnergy():

# calcualte the kinetic energy in joule
    ekin = 0
    i = 0
    mass = settings.mass
    while i < settings.nparticles:
        vx = settings.p[i].vx
        vy = settings.p[i].vy
        vz = settings.p[i].vz

        ekin += 0.5 * mass * (vx * vx + vy * vy + vz * vz)
        i += 1
    settings.Energy.ek = ekin
