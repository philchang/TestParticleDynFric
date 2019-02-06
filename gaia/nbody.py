import numpy as np
from scipy.integrate import ode
import argparse
import matplotlib.pyplot as pl
import math

import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gp
from galpy.util import bovy_conversion 
import scipy

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--show_plot', action='store_true', help='make plots (r.pdf, xy.pdf, z.pdf)')
parser.add_argument('--forward', action='store_true', help='forward in time integration')
parser.add_argument('--use_MW', action='store_true', help='MW Potential')
parser.add_argument('--use_MW14', action='store_true', help='MW14 Potential')
parser.add_argument('--use_K13', action='store_true', help='K13 NFW MW values')
parser.add_argument('--use_NFW', action='store_true', help='NFW Potential')
parser.add_argument('--use_Hernquist', action='store_true', help='NFW Potential')
parser.add_argument('--use_DF', action='store_true')

args = parser.parse_args()

useNFW = args.use_NFW or args.use_K13
useHernquist = args.use_Hernquist

MWpot = None
if args.use_MW : 
    MWpot = gp.MWPotential
elif args.use_MW14 : 
    MWpot = gp.MWPotential2014

pc = 3.0857e18
kpc = 1e3*pc
vkms = 1e5
msun = 1.989e33
G = 6.67e-8
m = 1.
TINY = 1e-30
yrs = 365.24*24*3600
Myrs = 1e6*yrs
Gyrs = 1e3*Myrs

Mvir = 1.5e12*msun
cvir =9.56
Rvir = 299.*kpc

msat = 1e10*msun

tEnd = 5.0*Gyrs

if args.use_K13 :
    # K13 values
    Rvir = 329.*kpc
    cvir = 9.36
    Mvir = 2.e12*msun

r0 = 8.0 # r0 8kpc
v0 = 220. # rotation rate

def derivs(t, y):
    
    arr = y.reshape(2,N,3)
    r = arr[0,:,:]
    v = arr[1,:,:]
    a, dt = calAcc( r, v)
    return np.array([v,a]).flatten()

def findTimeStep( r,v) : 
    a, dt = calAcc( r, v)
    return 0.1*dt.min()

def getICs() : # return ICs in galactocentric coordinates
    distance = 129.4e3*u.pc 

    c1 = coord.ICRS(ra=143.8868*u.degree, dec=-36.7673*u.degree,
                    distance=129.4e3*u.pc,
                    pm_ra_cosdec=-0.095*u.mas/u.yr,
                    pm_dec=0.058*u.mas/u.yr,
                    radial_velocity=290.7*u.km/u.s)

    gc1 = c1.transform_to(coord.Galactocentric)

    vx = gc1.v_x
    vy = gc1.v_y
    vz = gc1.v_z

    l = 264.8955
    b = 11.2479 
    R0 = r0*1.e3*u.pc

    #convert to Galactocentric -- 

    x = distance * np.cos(b*np.pi/180.) * np.cos(l*np.pi/180.) - R0
    y = distance * np.cos(b*np.pi/180.) * np.sin(l*np.pi/180.) 
    z = distance* np.sin(b*np.pi/180.)

    positions = np.array( [[x.value,y.value,z.value]])*pc
    velocities = np.array([[vx.value,vy.value,vz.value]])*vkms

    return positions,velocities

def NFWmass( r) : 
    scalelength = Rvir/cvir
    x = r/scalelength
    xRv = Rvir/scalelength
    mass0 = Mvir/(math.log(1. + cvir) - cvir/(1.+cvir))

    fcvir = math.log(1.+cvir)-(cvir/(1.+cvir))
    rhos = Mvir/(4*math.pi*(scalelength**3)*fcvir)

    #following Zentner & Bullock 03, section 2.2, eqn 3                                                                      

    Vvir = math.sqrt(G*Mvir/Rvir)

    Vmaxsq = 0.216*(Vvir**2)*(cvir/fcvir)
    Vmax = math.sqrt(Vmaxsq)

    rs = Rvir/cvir
    xsigma = r/rs  # Zentner & Bullock 2003, equation 6
   
    sigma = Vmax*((1.439*(xsigma**0.354))/(1.+1.1756*(xsigma**0.725)))

    if (r < Rvir) : 
        m = mass0*(math.log(1.+x) - (x/(1.+x)))
        rho = rhos/(x*((1.+x)**2))

    else : 
        m=(mass0*(math.log(1+xRv) - (xRv/(1.+xRv))))
        rho = 0.

    return m,rho,sigma

def hernquistmass( r, m0 = 1.29e12*msun, vc200=1.8e7, c0=9.39) : 
    r200 = G*m0/(vc200*vc200)
    rs = r200/c0
    a = rs*math.sqrt(2.*(math.log(1.+c0) - c0/(1.+c0)))
    m = m0 * r*r/((r+a)*(r+a))
    rho = (m0/(2.*math.pi))*(a/(r*(r+a)**3.))

    Vmax = 215.*vkms  # for V200 = 160                                                                                 
    rs = r200/c0
    x = r/rs  # Zentner & Bullock 2003, equation 6                                                                      
    sigma = Vmax*((1.439*(x**0.354))/(1.+1.1756*(x**0.725)))
    return m, rho, sigma

def calAcc(r, v) : 
    N = r.shape[0]
    a = np.zeros([N,3])
    dt_min = np.ones(N)*1e5*yrs
    dr = r
    dx = r[:,0]
    dy = r[:,1]
    dz = r[:,2]
    dv = v
    dr_gp = np.sqrt(dx*dx + dy*dy)/(r0*kpc) # galpy units
    dz_gp = dz/(r0*kpc)
    drNorm = np.linalg.norm(dr, axis = 1)
    acc = None
    if useNFW : 
        m, rho, sigma = NFWmass( drNorm)
        acc = -G*m/(drNorm*drNorm)[:,np.newaxis]*dr/drNorm[:,np.newaxis]
    elif useHernquist:
        m, rho, sigma = hernquistmass( drNorm)
        acc = -G*m/(drNorm*drNorm)[:,np.newaxis]*dr/drNorm[:,np.newaxis]
    else : 
        acc = np.zeros([dr_gp.size,3])
        conv = 1e-13*1e5
        for i, x_gp, y_gp, r_gp, z_gp in zip(range(dr_gp.size), dx/(r0*kpc), dy/(r0*kpc), dr_gp, dz_gp) : 
            a = gp.evaluateRforces(MWpot,r_gp,z_gp)*bovy_conversion.force_in_10m13kms2(v0,r0)*conv
            acc[i,0] = a*x_gp/r_gp
            acc[i,1] = a*y_gp/r_gp
            acc[i,2] = gp.evaluatezforces(MWpot,r_gp,z_gp)*bovy_conversion.force_in_10m13kms2(v0,r0)*conv
    dt_rmin = np.min(drNorm/np.linalg.norm(dv, axis=1))
    dt_amin = np.min(np.linalg.norm(dv,axis=1)/np.maximum(np.linalg.norm(acc,axis=1), TINY))
    dt_min = min(dt_rmin, dt_amin)
    if(args.use_DF) : 
        acc = acc + accDF( r, v)
    return acc, dt_min

def accDF( r, vel, msat=msat) : 
    if not (useNFW or useHernquist) :
        return np.zeros(r.shape)
    
    dr = r
    drNorm = np.linalg.norm(dr, axis = 1)
    
    m = 0. 
    rho = 0.
    sigma = 0.
    if useNFW : 
        m,rho,sigma = NFWmass( drNorm)
    elif useHernquist:
        m,rho,sigma = hernquistmass( drNorm)

    v = np.linalg.norm( vel, axis=1)

    xchandra = v/(math.sqrt(2.)*sigma)
    erf = scipy.special.erf(xchandra)
    # following Besla et al. 2007 here.  TODO: vary bmin as function of perturber mass.
    Lambda = drNorm/(3.*1.6*kpc)   
    #to avoid problem when r < Rsat
    logLambda=np.maximum(0.,math.log(Lambda))  

    accdf = (-4.*math.pi*G**2*(logLambda)*rho*(1./v**3)*(erf-2*xchandra*(1./math.sqrt(math.pi))*(math.exp(-(xchandra**2))))*msat)[:,np.newaxis]*vel
    return accdf

rp = kpc*np.arange( 2., 50., 1.)
m, rho, sigma = hernquistmass( rp)
rhoMean = m/(4.*math.pi/3.*rp**3)
rscale = 0.5*kpc
vKick = np.sqrt(G*rhoMean)*rscale 
for mass, r, v, rho in zip(m, rp, vKick, rhoMean) :
    print "{0:5.3e} {1:5.3e} {2:5.3e} {3:5.3e}".format( r/kpc, v/vkms, rho/msun*pc**3, r*(1e10*msun/mass)**0.333)

pl.plot( rp/kpc, rp*(1e10*msun/m)**0.333/kpc)
rs = np.arange( 0.1, 2, 0.1)*kpc
m, rho, sigma = hernquistmass( rs, m0=1.0e10*msun, vc200=38.7*vkms, c0=14.42)
print m/(4*3.141/3.*rs**3)/msun*pc**3
#pl.loglog( rs/kpc, m/msun)


#pl.loglog( rp/kpc, vKick/vkms)
pl.savefig("test.pdf") 
#print "rho = {0} {1} {2}".format(m/msun/(4.*math.pi/3.*(r/pc)**3), rho*pc**3/msun, m/msun/1e10)
pos, vel = getICs()
#print "Initial positions: {0} and velocities: {1}".format(pos/kpc, vel/vkms)
N = pos.shape[0]
if( not args.forward) : 
    tEnd = -tEnd
t = 0.
y0 = np.array([pos,vel]).flatten()
integral = ode( derivs).set_integrator("dop853")
integral.set_initial_value(y0, t)

tArray = []
rArray = []
xArr = []
yArr = []
zArr = []

while integral.successful() and ((not args.forward and tEnd < t) or (args.forward and t<tEnd)) :
    dt = 0.
    if( not args.forward) : 
        dt = max(-findTimeStep(pos,vel),tEnd-t)
    else :
        dt = min(findTimeStep(pos,vel),tEnd-t)

    y = integral.integrate( integral.t+dt)
    t = integral.t
    arr = y.reshape(2,N,3)
    pos = arr[0,:,:]
    vel = arr[1,:,:]
    r = np.linalg.norm( pos, axis=1)
    xArr.append(pos[0,0]/kpc)
    yArr.append(pos[0,1]/kpc)
    zArr.append(pos[0,2]/kpc)
    tArray.append(t/Gyrs)
    rArray.append(r[0]/kpc)

minr = np.array(rArray).argmin()
print "r_peri = {0} at t = {1}".format( rArray[minr], tArray[minr])
print "rdisk = {0} zdisk = {1}".format( np.sqrt(np.array(rArray)**2 - np.array(zArr)**2)[minr], zArr[minr])

if( args.show_plot) : 
    pl.plot( tArray, rArray)
    pl.xlabel( "t [Gyrs]")
    pl.ylabel( "r [kpc]")

    pl.savefig("r.pdf")
    pl.clf()
    pl.plot( xArr, yArr)
    pl.xlabel( "x [kpc]")
    pl.ylabel( "y [kpc]")

    pl.savefig("xy.pdf")
    pl.clf()
    pl.plot( np.sqrt(np.array(rArray)**2 - np.array(zArr)**2), zArr)
    pl.xlabel( "$r_{cyl}$ [kpc]")
    pl.ylabel( "z [kpc]")

    pl.savefig("z.pdf")


