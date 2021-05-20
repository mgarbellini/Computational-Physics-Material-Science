import settings
import numpy as np


def forceLJvec():


    xi=np.transpose(np.multiply(np.ones(shape=(settings.ntot,settings.ntot)),settings.xi))
    xit=np.transpose(xi)
    yi=np.transpose(np.multiply(np.ones(shape=(settings.ntot,settings.ntot)),settings.yi))
    yit=np.transpose(yi)
    zi=np.transpose(np.multiply(np.ones(shape=(settings.ntot,settings.ntot)),settings.zi))
    zit=np.transpose(zi)


    xij = xi - xit
    yij = yi - yit
    zij = zi - zit

    #PBC
    xijpbc = np.where(np.abs(xij) > 0.5*settings.box.lx, xij-np.sign(xij)*settings.box.lx, xij)
    yijpbc = np.where(np.abs(yij) > 0.5*settings.box.ly, yij-np.sign(yij)*settings.box.ly, yij)
    zijpbc = np.where(np.abs(zij) > 0.5*settings.box.lz, zij-np.sign(zij)*settings.box.lz, zij)


    R2 = np.multiply(xijpbc,xijpbc) + np.multiply(yijpbc,yijpbc) + np.multiply(zijpbc,zijpbc)



    R2 += 1.0e10*np.identity(settings.ntot)

    oneoverR2 = np.reciprocal(R2)
    rc2overR2 = settings.r0 *settings.r0 * oneoverR2 # r_c^2 /rij^2

    rc6overR6 = np.multiply(np.multiply(rc2overR2, rc2overR2),rc2overR2)

    aa = rc6overR6 - np.ones(shape=(settings.ntot,settings.ntot))

    fpot = 8.0*settings.eps*(np.multiply(rc6overR6, aa)) # need to add the shift of the potential

    fforce0 = 48.0*settings.eps * np.multiply(rc6overR6, rc6overR6 - 0.5*np.ones(shape=(settings.ntot, settings.ntot)))
    fforce = np.multiply(fforce0, oneoverR2)

    fforcecut = np.where(R2 > settings.cutoff*settings.cutoff, np.zeros(shape=(settings.ntot, settings.ntot)), fforce)
    epotij = np.where(R2 > settings.cutoff*settings.cutoff, np.zeros(shape=(settings.ntot, settings.ntot)), fpot)
    eptiju = np.triu(epotij, 0)



    settings.Energy.ep = np.sum(eptiju)


    # fij along x
    fijx = -np.multiply(fforcecut, xijpbc)
    fijy = -np.multiply(fforcecut, yijpbc)
    fijz = -np.multiply(fforcecut, zijpbc)


    # total force on atoms i
    settings.sumfix = np.sum(fijx,axis=0)
    settings.sumfiy = np.sum(fijy,axis=0)
    settings.sumfiz = np.sum(fijz,axis=0)




def forceLJ():

    epot = 0.
    for i in range(0,settings.nparticles):
        settings.p[i].fx = 0.
        settings.p[i].fy = 0.
        settings.p[i].fz = 0.

    i = 0
    sf2a = settings.r0*settings.r0 / settings.cutoff / settings.cutoff
    sf6a = sf2a * sf2a * sf2a
    epotcut = 8.*settings.eps*sf6a*(sf6a - 1.)

    while i < settings.nparticles - 1:
        j = i + 1
        while j < settings.nparticles:
            rijx = pbc(i,j,1)
            rijy = pbc(i,j,2)
            rijz = pbc(i,j,3)
            r2 = rijx * rijx + rijy * rijy + rijz * rijz
            # calculate fx, fy, fz
            if r2 < settings.cutoff * settings.cutoff:
                sf2 = settings.r0*settings.r0 / r2
                sf6 = sf2 * sf2 * sf2
                epot += (8.*settings.eps*sf6*(sf6 - 1.)) #-epotcut)
                ff = 48.*settings.eps*sf6*(sf6 - 0.5)/r2
                settings.p[i].fx -= ff*rijx
                settings.p[i].fy -= ff*rijy
                settings.p[i].fz -= ff*rijz
                settings.p[j].fx += ff*rijx
                settings.p[j].fy += ff*rijy
                settings.p[j].fz += ff*rijz
            j += 1
        i += 1

    settings.Energy.ep = epot
    #for i in range(0,settings.nparticles):
    #    print(i, settings.p[i].fx, settings.p[i].fy, settings.p[i].fz, settings.p[i].vx, settings.p[i].vy, settings.p[i].vz)


def pbc(i,j,k):


    if k == 1:
        xi = settings.p[i].x
        xj = settings.p[j].x
        l = settings.box.lx
    elif k == 2:
        xi = settings.p[i].y
        xj = settings.p[j].y
        l = settings.box.ly
    elif k == 3:
        xi = settings.p[i].z
        xj = settings.p[j].z
        l = settings.box.lz

    xi = xi % l
    xj = xj % l

    rij = xj - xi
    if abs(rij) > 0.5*l:
        rij = rij - np.sign(rij) * l

    return rij
