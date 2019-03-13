import numpy as np
from scipy.integrate import ode
import argparse
import math

try :
    import matplotlib
    matplotlib.use("Agg")
except ImportError : 
    print( "Running without matplotlib")

import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gp
from galpy.util import bovy_conversion 
import scipy
import sys 


useMPI = True
try :
    from mpi4py import MPI
except ImportError:
    useMPI = False

rank = 0
size = 1
comm = None
if useMPI : 
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2 : 
        useMPI = False
    else : 
        if rank == 0 :
	    print( "Running with MPI") 

#from numba import jit 

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('--show_plot', action='store_true', help='make plots (r.pdf, xy.pdf, z.pdf)')
#parser.add_argument('--forward', action='store_true', help='forward in time integration')
parser.add_argument('--use_MW', action='store_true', help='MW Potential')
parser.add_argument('--use_MW14', action='store_true', help='MW14 Potential')
parser.add_argument('--use_K13', action='store_true', help='K13 NFW MW values')
parser.add_argument('--use_NFW', action='store_true', help='NFW Potential')
parser.add_argument('--use_Hernquist', action='store_true', help='NFW Potential')
parser.add_argument('--use_DF', action='store_true')
parser.add_argument('--use_TestParticles', action='store_true')
parser.add_argument('--run_test', action='store_true')
args = parser.parse_args()

useNFW = args.use_NFW or args.use_K13
useHernquist = args.use_Hernquist
useDF = args.use_DF
useTestParticles = args.use_TestParticles

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

# MW parameters
Mvir = 1.5e12*msun
cvir = 9.56
Rvir = 299.*kpc

# for hernquist
M0 = 1.29e12*msun
vc200=1.8e7
c0=9.39

# satellite parameters for hernquist
Nsat = 1
msat = np.zeros(Nsat)
c0sat = np.zeros(Nsat)
msat[0] = 1e10*msun
vc200sat = vc200*(msat/M0)**0.333e0
c0sat[:] = 9.39 

# test Particle parameters
nDisk = 1000000
rbins = 500

tEnd = 0.5*Gyrs # in the future
tStart = -0.5*Gyrs # in the past
tStep = 0.025*Gyrs

if args.use_K13 :
    # K13 values
    Rvir = 329.*kpc
    cvir = 9.36
    Mvir = 2.e12*msun

r0 = 8.0 # r0 8kpc
v0 = 220. # rotation rate

def derivs(t, y):
    N = y.size/6
    arr = y.reshape(2,N,3)
    r = arr[0,:,:]
    v = arr[1,:,:]
    a, dt = calAcc( r, v)
    return np.array([v,a]).flatten()

def findTimeStep( r,v) : 
    a, dt = calAcc( r, v)
    return 0.2*dt.min()

def getTestICs() : 
    x = 0.
    y = 5.0*kpc
    z = 0.

    vx = -500.*vkms
    vy = 0.
    vz = 0.

    positions = np.array( [[[x,y,z]]])
    velocities = np.array([[[vx,vy,vz]]])
    return positions,velocities


def getICs(pm_ra_cosdec=-0.095,pm_dec=0.058,radial_velocity=290.7) : # return ICs in galactocentric coordinates
    distance = 129.4e3*u.pc 

    c1 = coord.ICRS(ra=143.8868*u.degree, dec=-36.7673*u.degree,
                    distance=129.4e3*u.pc,
                    pm_ra_cosdec=pm_ra_cosdec*u.mas/u.yr,
                    pm_dec=pm_dec*u.mas/u.yr,
                    radial_velocity=radial_velocity*u.km/u.s)

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

def getVelSamples( nn=10) : 

    vavg = [-80.93960285062732,-48.51070213345165, 50.56259604843662]
    sigmax = 22.4883097
    sigmay = 231.308893
    sigmaz = 64.9421574

    svx = np.random.normal(vavg[0], sigmax, nn)
    svy = np.random.normal(vavg[1], sigmay, nn)
    svz = np.random.normal(vavg[2], sigmaz, nn)

    velocities = np.array(list(zip(svx,svy,svz)))*vkms
    return velocities

def getICsmusample(nn=10):
    distance = 129.4e3*u.pc

# define pm samples:
    spmracosdec = np.random.normal(-0.095, 0.018, nn)
    spmdec = np.random.normal(0.058, 0.024, nn)

    vx = np.empty(nn)
    vy = np.empty(nn)
    vz = np.empty(nn)
    positions = []
    velocities = []
    for pm_ra_cosdec, pm_dec in zip( spmracosdec, spmdec):
        pos, vel = getICs(pm_ra_cosdec=pm_ra_cosdec,pm_dec=pm_dec)
        positions.append( pos)
        velocities.append( vel)

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

def hernquistmass( r, m0 = M0, vc200=vc200, c0=c0) : 
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
        acc = (-G*m/(drNorm*drNorm*drNorm))[:,np.newaxis]*dr#[:,np.newaxis]*dr
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
    if(useDF) : 
        acc[:Nsat,:] = acc[:Nsat,:] + accDF( r[:Nsat,:], v[:Nsat,:])
    if(useTestParticles and Nsat < N) :
        # include accelerations from other satellites for test particles
        for i in range(Nsat) : 
            rsat = r[i]
            dr = r[Nsat:,:] - rsat[np.newaxis,:]
            drNorm = np.linalg.norm( dr, axis=1)
            m, rho, sigma = hernquistmass( drNorm, m0=msat[i], vc200=vc200sat[i], c0=c0sat[i])
            acc[Nsat:,:] =  acc[Nsat:,:] - (G*m/(drNorm*drNorm*drNorm))[:,np.newaxis]*dr
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

def getStartingPosition(pos, vel, tEnd = tStart) :
    print( "Initial positions: {0} and velocities: {1}".format(pos/kpc, vel/vkms))
    N = pos.shape[0]

    t = 0.
    y0 = np.array([pos,vel]).flatten()
    integral = ode( derivs).set_integrator("dop853")
    integral.set_initial_value(y0, t)

    tArray = []
    rArray = []
    xArr = []
    yArr = []
    zArr = []

    while integral.successful() and tEnd < t :
        dt = max(-findTimeStep(pos,vel),tEnd-t)
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
    print( "Ending positions: {0} and velocities: {1}".format(pos/kpc, vel/vkms))

    #print ("r_peri = {0} at t = {1}".format( rArray[minr], tArray[minr]))
    #print ("rdisk = {0} zdisk = {1}".format( np.sqrt(np.array(rArray)**2 - np.array(zArr)**2)[minr], zArr[minr]))
    return t, pos,vel

def setupDiskParticles( nDisk=nDisk, rbins=rbins, rin = 2.*kpc, rout=30.*kpc) :
    thetaBins = int(nDisk/rbins)
    dr = (rout-rin)/rbins
    r = np.arange(rin, rout, dr)
    m, rho, sigma = hernquistmass(r)
    theta = np.arange(0., 2.*math.pi, 2.*math.pi/thetaBins)
    phase = 2.*math.pi*np.random.rand(rbins)
    numParticles = theta.size*r.size
    pos = np.zeros( [numParticles, 3])
    vel = np.zeros( [numParticles, 3])
    for i in range(rbins) : 
        cosx = np.cos( theta + phase[i])
        sinx = np.sin( theta + phase[i])
        pos[i*thetaBins:(i+1)*thetaBins,0] = r[i]*cosx
        pos[i*thetaBins:(i+1)*thetaBins,1] = r[i]*sinx
        v = math.sqrt(G*m[i]/(r[i]))
        vel[i*thetaBins:(i+1)*thetaBins,0] = -v*sinx
        vel[i*thetaBins:(i+1)*thetaBins,1] = v*cosx
    
    return pos, vel


rp = kpc*np.arange( 10., 50., 1.)
m, rho, sigma = hernquistmass( rp)
rhoMean = m/(4.*math.pi/3.*rp**3)
rscale = 0.5*kpc
vKick = np.sqrt(G*rhoMean)*rscale 
posSamples, velSamples = getICsmusample( nn = 1)
if( args.run_test) :
    posSamples, velSamples = getTestICs()
#posSamples, velSamples = getICsmusample( nn = 1)
#vdist = getVelSamples()

for posSat, velSat in zip( posSamples, velSamples) : 
    #print ("pos,vel")
    #print (pos, vel)
    # we have run it backwards -- now run forward
    t = 0.
    posSatInit = None
    velSatInit = None
    if( rank == 0) : 
        t, posSatInit, velSatInit = getStartingPosition( posSat, velSat)
    
    if( useTestParticles) : 
        if( useMPI) : 
            data = None
            if rank == 0:
                data = np.array([posSatInit,velSatInit])

            data = comm.bcast(data, root=0) # broadcast the array from rank 0 to all others
        
            posSatInit = data[0]
            velSatInit = data[1]

        dt = 0

        posDisk, velDisk = setupDiskParticles()

        Ntest = 0
        if useMPI : 
            if rank == 0:
                Ntest = posDisk.shape[0]
                posDisk = posDisk.reshape(size,Ntest/size,3)
                velDisk = velDisk.reshape(size,Ntest/size,3)
                
            posDisk = comm.scatter(posDisk, root=0)
            velDisk = comm.scatter(velDisk, root=0)

        pos = np.append( posSatInit, posDisk,axis=0)
        vel = np.append( velSatInit, velDisk,axis=0)
        t = 0. # needs to be zeroed again
        N = pos.shape[0] # reset to include test particles
        y0 = np.array([pos,vel]).flatten()
    
        integral = ode( derivs).set_integrator("dop853")
        integral.set_initial_value(y0, t)
        iFrame = 0
        while t < tEnd-tStart :
            tNext = t + tStep
            from timeit import default_timer as timer
            start = timer()
            while integral.successful() and t < tNext : 
                dt = min(findTimeStep(pos,vel),tNext-t)
                #print( "t={0:.3f} {1:.3e}".format(integral.t/Gyrs, dt/Gyrs))
                y = integral.integrate( integral.t+dt)
                t = integral.t
                arr = y.reshape(2,N,3)
                pos = arr[0,:,:]
                vel = arr[1,:,:]
                posSat = pos[:Nsat]
                velSat = vel[:Nsat]
                r = np.linalg.norm( posSat, axis=1)
                #print( "r={0:.3f}".format(r[0]/kpc))
            
            if useMPI : 
                data1 = comm.gather(pos[Nsat:,:], root=0)
                data2 = comm.gather(vel[Nsat:,:], root=0)
                if( rank == 0 ) :
                    pos = np.append(posSat,np.array(data1).reshape(Ntest,3),axis=0)
                    vel = np.append(velSat,np.array(data2).reshape(Ntest,3),axis=0)

            if( args.show_plot and rank == 0) : 
                import matplotlib.pyplot as pl
                from scipy.stats.kde import gaussian_kde
                posTest = pos[Nsat:]
                velTest = vel[Nsat:]
                xDisk = posTest[:,0]/kpc
                yDisk = posTest[:,1]/kpc
                xSat = posSat[:,0]/kpc
                ySat = posSat[:,1]/kpc

                pl.clf()
                iFrame = iFrame + 1
                fig = pl.figure()
                ax = fig.add_subplot(111)
                ax.set_aspect('equal')
                ax.scatter( xDisk, yDisk, s=1)
                ax.scatter( xSat, ySat, s=10, color="red")
                pl.xlabel( "x [kpc]", fontsize=20)
                pl.ylabel( "y [kpc]", fontsize=20)
                ax.set_xlim(-35,35)
                ax.set_ylim(-35,35)
                ax.text(8,31, "t={0:.3f} Gyrs".format((t+tStart)/Gyrs), fontsize=14)
                plotName = "frame{0:04d}.png".format(iFrame)
                print( "making plot: {0}".format(plotName))
                pl.savefig( plotName)
                pl.close(fig)
            end = timer()
            if rank == 0 :
                print(end - start) # Time in seconds, e.g. 5.38091952400282



