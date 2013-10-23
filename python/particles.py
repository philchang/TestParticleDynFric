import math
import numpy
import constants


class Test_particle :
    def __init__(self, pos, vel) :
        self.pos = pos
        self.vel = vel

    def update( self, pos, vel) :
        self.pos = pos
        self.vel = vel

class Perturber(Test_particle) :
    
    def __init__(self, pos, vel, profile, has_df = False) :
        self.pos = pos
        self.vel = vel
        self.profile = profile
        self.has_df = has_df

    def get_radius( self, pos) :
        newPos = pos-self.pos
        return math.sqrt(numpy.dot(newPos,newPos))
        
    def get_acceleration( self, pos, vel, mass) :
        newPos = pos-self.pos
        newVel = vel-self.vel
        r = math.sqrt(numpy.dot(newPos,newPos))
        dmMass = self.profile.get_mass(r)
        acc = -constants.G_newton * dmMass*newPos/(r*r*r)

        if (self.has_df) : # include dyn friction
            v = math.sqrt(numpy.dot(newVel, newVel))
            sigma = self.profile.get_sigma(r)
            rho = self.profile.get_rho(r)
            xchandra = v/(math.sqrt(2.)*sigma)
            erf = math.erf(xchandra)
            Lambda = r/(3.*1.6*constants.Kpc)    # following Besla et al. 2007 here.  TODO: vary bmin as function of perturber mass.
            logLambda = max(0.0, math.log(Lambda))  # to avoid problem when r < Rsat

            accdf = -4.*math.pi*constants.G_newton**2.*logLambda*rho/(v**3.)*(
                erf-2.*(xchandra/math.sqrt(math.pi))*(math.exp(-(xchandra**2.))))*mass*newVel

            acc += accdf

        return acc 

    def write_out(self) :
        print str(self.profile.get_mass()/constants.Msun) + " " + str(self.pos/constants.Kpc) + " " + str(self.vel/constants.Kms)

