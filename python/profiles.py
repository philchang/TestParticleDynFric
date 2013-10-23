import math
import constants
import numpy

class Hernquist_profile :
    def __init__(self, m0, v200, conc) :
        self.m0 = m0
        self.v200 = v200
        self.conc = conc

    def get_mass( self, r = None) :
        if( r == None) :
            return self.m0

        r200 = constants.G_newton*self.m0/(self.v200*self.v200)
        rs = r200/self.conc
        a = rs*math.sqrt(2.*(math.log(1.+self.conc) - self.conc/(1.+self.conc)))
        mass = self.m0 * r*r/(r+a)**2.
        return mass

    def get_sigma(self, r) :
        pass
    
    def get_rho( self, r) :
        pass
    
class NFW_profile :
     
    def __init__(self, mvir, rvir, cvir) :
        self.mvir = mvir
        self.rvir = rvir
        self.cvir = cvir

    def get_mass( self, r = None) :
        # NFW mass
        if( r == None) :
            return self.mvir
        
        mass0 = self.mvir/(math.log(1e0 + self.cvir) - self.cvir/(1e0+self.cvir))
        scalelength = self.rvir/self.cvir

        x = r/scalelength
        if (r <= self.rvir) :
            return mass0*(math.log(1.+x) - (x/(1.+x)))
        else :
            return self.mvir

    def get_sigma(self, r) :
        # following Zentner & Bullock 03, section 2.2, eqn 3                                                                      

        Vvir = math.sqrt(constants.G_newton*self.mvir/self.rvir)
        gcvir = math.log(1.+self.cvir)-(self.cvir/(1.+self.cvir))
        Vmax = math.sqrt(0.216*(Vvir*Vvir)*(self.cvir/gcvir))
        rs = self.rvir/self.cvir
        x = r/rs  # Zentner & Bullock 2003, equation 6
        sigma = Vmax*((1.439*(x**0.354))/(1.+1.1756*(x**0.725)))

        return sigma

    def get_rho( self, r) :
        fcvir = math.log(1.+self.cvir)-(self.cvir/(1.+self.cvir))
        scalelength = self.rvir/self.cvir
        rhos = self.mvir/(4*math.pi*(scalelength**3)*fcvir)
        x = r/scalelength
        if (r <= self.rvir) :
            rho = rhos/(x*((1.+x)**2.))
        else : 
            rho = 0.

        return rho
