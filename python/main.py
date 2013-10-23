import numpy
import math
import calc_orbit
import particles
import constants
import profiles
import sys
import matplotlib.pyplot as pl

if (len(sys.argv) != 2) :
    print "Usage python main.py <input file>" 
    exit(1)

names = numpy.loadtxt( sys.argv[1], skiprows=1, usecols=[0], dtype="string")
arr = numpy.loadtxt( sys.argv[1], skiprows=1, usecols=range(1,8))

masses = arr[:,0]*constants.Msun
v200s = 220.*(masses/(1e12*constants.Msun))**0.333*constants.Kms
conc = 0.971 - 0.094*numpy.log(masses/(1.e12*constants.Msun))
concentrations = 10.**conc
positions = arr[:,1:4]*constants.Kpc
velocities = arr[:,4:7]*constants.Kms

# add the central galaxy
MWmvir = 1.5e12*constants.Msun
MWrvir = 299e0*constants.Kpc
MWcvir = 9.56

#MWmvir = 2e12*constants.Msun
#MWrvir = 329e0*constants.Kpc
#MWcvir = 9.36

MWpos = numpy.zeros(3)
MWvel = numpy.zeros(3)
MWgalaxy = particles.Perturber( MWpos, MWvel, profiles.NFW_profile( MWmvir, MWrvir, MWcvir), has_df = True)

orbits = calc_orbit.Orbits()

orbits.add_perturber( MWgalaxy)

perturber_conc = 9.58
for mass, pos, vel, v200, conc in zip( masses, positions, velocities, v200s, concentrations) :
    orbits.add_perturber( particles.Perturber(pos, vel, profiles.Hernquist_profile(mass, v200,  conc)))

for perturber in orbits.perturbers :
    perturber.write_out()

orbits.set_current_time( 0.0)

dt = -10.*constants.Myr

# integrate backwards
for t in numpy.arange( dt, -10.*constants.Gyr, dt) :
    orbits.evolve( dt)
    print "t = " + str(orbits.currentTime/constants.Gyr) 

dt = 10.*constants.Myr

perturber_radii = []
times = []
central_galaxy = orbits.perturbers[0]
LMC = orbits.perturbers[1]
SMC = orbits.perturbers[2]

# integrate forward
for t in numpy.arange( orbits.currentTime, 0.0, dt) :
    orbits.evolve( dt)
    radii = []
    for perturber in orbits.perturbers :
        pos = perturber.pos
        radii.append( central_galaxy.get_radius(pos)/constants.Kpc)

    #radii.append(LMC.get_radius(SMC.pos)/constants.Kpc)
    times.append(orbits.currentTime/constants.Gyr)
    perturber_radii.append( radii) 
    
    print "t = " + str(orbits.currentTime/constants.Gyr) + " Gyrs"

#names = numpy.append(names, "LMC-SMC")
perturber_radii = numpy.array(perturber_radii)
pert_radii = perturber_radii[:,1:]
for i in range(pert_radii[0,:].size) :
    pl.plot( times, pert_radii[:,i], label=names[i])
    

pl.legend()
pl.savefig('test.pdf')


for perturber in orbits.perturbers :
    perturber.write_out()
