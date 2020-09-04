try :
    import mkl
    mkl.set_num_threads(4)
except :
    print("Not using intel/mkl")

import numpy as np
import scipy as sp
import math
import emcee
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": True})

import matplotlib.pyplot as pl
import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gp
from galpy.potential import MWPotential2014
from galpy.potential import evaluatePotentials
from galpy.potential import evaluateRforces
from galpy.potential import evaluatezforces
from galpy.util import bovy_conversion
from galpy.potential import DehnenBarPotential
from galpy.potential import HernquistPotential
import matplotlib.pyplot as pl

filename = "mcmc.h5" #default file, override with -o


iterations = 12000
thin = 200
discard =3000
unique_name = True

TINY_GR = 1e-20
multithread = False
processes = None

excluded_pulsars = None
nanograv = [0,1,2,3]
ppta = [4,5,6,7,8,9]
epta = [10,11,12,13]
best_pulsars = [0, 12, 2, 3, 4, 5, 6, 9, 10, 11]

#use_other_distance = [5, 7, 9]
use_other_distance = [5,7,9,14,17]
#best_pulsars = nanograv
pulsars_number = None
pulsar_data = None

# model
QUILLEN = 0
EXPONENTIAL = 1
GAUSSIAN = 2
MWpot = 3
Hernquistfix = 4
HERNQUIST = 5
HALODISK = 6
LOCAL = 7
SECH2 = 8
POWER_LAW = 9
QUILLENBETA = 10
BULGE = 11

number_parameters = 0 # number of parameters for the galactic model

DEFAULT_MODEL = QUILLEN
MODEL = DEFAULT_MODEL

def initialize_model( model=MODEL) : 
    global number_parameters, MODEL
    if model == QUILLEN : 
        number_parameters = 2
    elif MODEL == QUILLENBETA:
        number_parameters = 2  ## this is alpha1 and beta only 
    elif model == EXPONENTIAL or model == GAUSSIAN or model == SECH2:
        number_parameters = 2
    elif model == MWpot :
        number_parameters = 2
    elif model == Hernquistfix: 
        number_parameters = 0
    elif model == HERNQUIST:
        number_parameters = 2
    elif model == HALODISK: 
        number_parameters = 4
    elif model == LOCAL : 
        number_parameters = 3
    if model == POWER_LAW : 
        number_parameters = 2
    elif MODEL == BULGE:
        number_parameters = 2

    MODEL = model
    print ("MODEL=",MODEL)

## location of Sun -
rsun= 8.122 ## in kpc
xsun = rsun
ysun = 0.
zsun = 0.0055  ## in kpc from Quillen et al. 2020
phisun = np.arctan2(xsun, ysun)

kpctocm = 3.086e21 ## convert pc to cm
pc = kpctocm*1e-3
Msun = 1.989e33   ## in g

day_to_sec = 24*3600
c = 3e10  ## in cm 

#Quillen model constants
alpha1_0 = 4e-30
alpha2_0 = -1e-51
beta0 = 0.
#beta0 = 0.
brange = 0.5 ## difference between Li et al. 2019, Mroz et al. 2019 and beta = -0.05 adopted in Quillen
lgalpha1_0 = math.log10( alpha1_0)
lgalpha2_0 = math.log10( -alpha2_0)

p1_range = 3
p2_range = 2
Mhrange = 0.86169 ## from Xue et al. 2008 and Boylan-Kolchin et al. 2013

# exponential or gaussian model constants
rho_midplane = 0.1 # solar mass per cubic parsec
rho_midplane *= 2e33/(pc**3)
lgrho_midplane = math.log10(rho_midplane)
scale_height = 0.1*kpctocm # 500 pc
lgscale_height = math.log10(scale_height)

## Hernquist model 
Mh_0 = 1.e12*Msun ## in g 
a_0 = 30.*kpctocm  ## in cm
lgMh_0 = math.log10(Mh_0)
lga_0 = math.log10(a_0)

xsun *= kpctocm
ysun *= kpctocm
zsun *= kpctocm

mas_per_year_to_as_per_sec = 1e-3/3.15e7

Vlsr_quillen = 233.e5 ## from Schonrich et al. 2010 in cm/s

Vlsr = 255.2*1e5
#Vlsr = Vlsr_quillen
Vlsr0 = Vlsr_quillen
Vlsr_err = 1.4*1e5  ## reflecting Schonrich et al. 2010 
G = 6.67e-8

# MW 2014 parameters
alpha_mw2014 = 1.8
alpha_mw2014_range = 0.1
rc_mw2014 = 1.9
rc_mw2014_range = 0.1
lg_a0_mw2014 = math.log(3.)
lg_a0_mw2014_range = 0.5
lg_b0_mw2014 = math.log(0.28)
lg_b0_mw2014_range = 0.5

# Local expansion
dadr0 = -1
dadr_range = 1
dadphi0 = 0.
dadphi_range = 1
lgdadz0 = lgalpha1_0
lgdadz_range = p1_range

# power law
eta0 = 0
eta_range = 0.9
zmid0 = 0
zmid_range = 0.1

def accHernquistfix(x,y,z):
    x = x/kpctocm
    y = y/kpctocm
    z = z/kpctocm
    r = np.sqrt( x**2 + y**2)
    pot = HernquistPotential(2.e12*u.M_sun , 30e3*u.pc)
    az = evaluatezforces(pot,r*u.kpc , z*u.kpc)
    ar = evaluateRforces(pot,r*u.kpc , z*u.kpc)
    az = -az*1.e5/(1.e6*3.15e7)
    ar = ar*1.e5/(1.e6*3.15e7)
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az


def accMWpot(x,y,z,alpha=1.8, rc=1.9, a=3., b=0.28):
    x = x/kpctocm
    y = y/kpctocm
    z = z/kpctocm
    r = np.sqrt( x**2 + y**2)
    bp= gp.PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
    mp= gp.MiyamotoNagaiPotential(a=a/8.,b=b/8.,normalize=.6)
    nfw= gp.NFWPotential(a=16/8.,normalize=.35)
    #print(help(MWPotential2014))
    pot = [bp,mp,nfw]
    #pot = MWPotential2014
    az = evaluatezforces(pot,r*u.kpc , z*u.kpc)*bovy_conversion.force_in_kmsMyr(Vlsr/1e5,8.122)
    ar = evaluateRforces(pot,r*u.kpc , z*u.kpc)*bovy_conversion.force_in_kmsMyr(Vlsr/1e5,8.122)
    ar = ar*1.e5/(1.e6*3.15e7)
    az = -az*1.e5/(1.e6*3.15e7)

    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az

def acc_gauss(x,y,z,rho0,z0, Vlsr) : 
    r = np.sqrt(x*x + y*y) 
    az = -4*np.pi*G*rho0*z0*sp.special.erf(np.abs(z)/z0)*np.sign(z)*math.sqrt(math.pi)*0.5
    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az

def acc_exp(x,y,z,rho0,z0, Vlsr, useSech2=False) :
#    rho0 = 1e1**lgrho0
#    z0 = 1e1**lgz0
    r = np.sqrt(x*x + y*y) 
    az = -4*np.pi*G*rho0*z0*(1.-np.exp(-np.abs(z)/z0))*np.sign(z)
    if( useSech2) :
        az = -4*np.pi*G*rho0*z0*np.tanh(z/z0)

    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az

def acc_quillenbeta(x,y,z,lgalpha1,beta,Vlsr):
    r = np.sqrt(x*x + y*y)
    alpha1 = 1e1**lgalpha1
    alpha2 = 0.

    az = -alpha1*z - alpha2*(z*z)*np.sign(z)
    ar = Vlsr*Vlsr/r

    if (number_parameters > 1):
        ar = (Vlsr**2)*((1./rsun)**(2.*beta))*(r**((2.*beta)-1.))

    ax = -ar*x/r
    ay = -ar*y/r

    return ax,ay,az     

def acc_quillen(x,y,z,lgalpha1, lgalpha2, Vlsr) :
    r = np.sqrt(x*x + y*y) 
    alpha1 = 1e1**lgalpha1
    alpha2 = 0.
    
    if (number_parameters > 1) : 
        alpha2 = -1e1**lgalpha2
    az = -alpha1*z - alpha2*(z*z)*np.sign(z)
    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r

    return ax,ay,az

def acc_power_law(x,y,z,lgalpha1, eta, zmid, Vlsr) :
    r = np.sqrt(x*x + y*y) 
    alpha1 = 1e1**lgalpha1
    az = -alpha1*(1 - eta*np.abs(z/kpctocm-zmid)) * (z-zmid*kpctocm)
    ar = Vlsr*Vlsr/r
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az


def acc_local_expansion( x, y, z, dadr=dadphi0, dadphi=dadphi0, dadz=1e1**lgdadz0) :
    r = np.sqrt(x*x + y*y)
    rsun = np.sqrt(xsun*xsun + ysun*ysun)

    arsun = Vlsr*Vlsr/rsun
    #ar = Vlsr*Vlsr/r
    ar = arsun*(1+dadr*(r-rsun)/rsun)
    deltaphi = np.arctan2(x, y) - phisun 
    aphi = arsun*dadphi*deltaphi
    az = -dadz*z
    ax = -ar*x/r - aphi*y/r
    ay = -ar*y/r + aphi*x/r
    return ax,ay,az

def accHernquist(x,y,z,lgMh,lga):
    r = np.sqrt(x**2 + y**2 + z**2)
    Mh = 1e1**lgMh
    a = 1e1**lga
    ar = (G*Mh)/((r+a)**2)
    az = -((G*Mh)*(z/r))/((r+a)**2)
    ax = -ar*x/r
    ay = -ar*y/r
    return ax,ay,az

def accHernquistplusdisk(x,y,z,lgMh,lga,rho0,z0, Vlsr):
    r = np.sqrt(x**2 + y**2 + z**2)
    Mh = 1e1**lgMh
    a = 1e1**lga
    arh = (G*Mh)/((r+a)**2)
    azh = ((G*Mh)*(z/r))/((r+a)**2)

## convert loga and logrho?

    rr = np.sqrt(x*x + y*y)
    azdisk = 4*np.pi*G*rho0*z0*(1.-np.exp(-np.abs(z)/z0))*np.sign(z)
    ardisk = Vlsr*Vlsr/rr

    arh *= (rr/r)

## is this correct? (r = sqrt(x^2+y^2) for disk but not for sphere) 
    ar = arh + ardisk
    az = -azh - azdisk 
    ax = -ar*x/rr
    ay = -ar*y/rr

    return ax,ay,az

## define acceleration components to fit the data, from Quillen et al. 2020:
def alos(x,y,z,parameters):
    global Vlsr
    ax,ay,az = 0,0,0
    axsun, aysun, azsun = 0,0,0
    if MODEL == QUILLEN : 
        lgalpha1, lgalpha2 = parameters[0], -np.inf
        if(number_parameters > 1) : 
            lgalpha2 = parameters[1]
        axsun, aysun, azsun = acc_quillen(xsun,ysun,zsun, lgalpha1, lgalpha2, Vlsr)
        ax, ay, az = acc_quillen(x,y,z,lgalpha1,lgalpha2, Vlsr)
    elif MODEL == QUILLENBETA : 
        lgalpha1, beta = parameters[0],-np.inf
        if(number_parameters > 1):
            beta = parameters[1]
        axsun, aysun, azsun = acc_quillenbeta(xsun,ysun,zsun, lgalpha1, beta, Vlsr)
        ax, ay, az = acc_quillenbeta(x,y,z,lgalpha1,beta, Vlsr)   
    elif MODEL == EXPONENTIAL or MODEL == SECH2:
        lgrho0, lgz0 = parameters[0:2]
        rho0 = 1e1**lgrho0
        z0 = 1e1**lgz0
        axsun, aysun, azsun = acc_exp(xsun,ysun,zsun, rho0, z0, Vlsr)
        ax, ay, az = acc_exp(x,y,z,rho0,z0, Vlsr)
    elif MODEL == GAUSSIAN :
        lgrho0, lgz0 = parameters[0:2]
        rho0 = 1e1**lgrho0
        z0 = 1e1**lgz0
        axsun, aysun, azsun = acc_gauss(xsun,ysun,zsun, rho0, z0, Vlsr)
        ax, ay, az = acc_gauss(x,y,z,rho0,z0, Vlsr)
    elif MODEL == MWpot:
        if number_parameters == 0 :
            axsun,aysun,azsun = accMWpot(xsun,ysun,zsun)
            ax,ay,az = accMWpot(x,y,z)
        else : 
            lga, lgb = parameters[0:2]
            axsun,aysun,azsun = accMWpot(xsun,ysun,zsun, a=1e1**lga, b=1e1**lgb)
            ax,ay,az = accMWpot(x,y,z, a=1e1**lga, b=1e1**lgb)

    elif MODEL == Hernquistfix:
        axsun,aysun,azsun = accHernquistfix(x,y,z)
        ax,ay,az = accHernquistfix(xsun,ysun,zsun)
    elif MODEL == HERNQUIST:
        lgMh, lga = parameters[0:2]
        Mh = 1e1**lgMh
        a = 1e1**lga
        axsun,aysun,azsun = accHernquist(x,y,z,lgMh,lga)
        ax,ay,az = accHernquist(xsun,ysun,zsun,lgMh,lga)
    elif MODEL == HALODISK:
        lgMh,lga,lgrho0,lgz0 = parameters[0:4]
        Mh = 1e1**lgMh
        a = 1e1**lga
        rho0 = 1e1**lgrho0
        z0 = 1e1**lgz0
        axsun,aysun,azsun = accHernquistplusdisk(xsun,ysun,zsun,lgMh,lga,rho0,z0,Vlsr)
        ax,ay,az = accHernquistplusdisk(x,y,z,lgMh,lga,rho0,z0,Vlsr)
    elif MODEL == LOCAL : 
        dadr, dadphi, lgdadz = parameters[0:3]
        dadz = 1e1**lgdadz
        axsun, aysun, azsun = acc_local_expansion( xsun, ysun, zsun, dadr, dadphi, dadz) 
        ax, ay, az = acc_local_expansion( x, y, z, dadr, dadphi, dadz) 
    if MODEL == POWER_LAW : 
        lgalpha1, zmid = parameters[0:2]
        eta = 0
        axsun, aysun, azsun = acc_power_law(xsun,ysun,zsun, lgalpha1, eta, zmid, Vlsr)
        ax, ay, az = acc_power_law(x,y,z,lgalpha1, eta, zmid, Vlsr)
    
    dx, dy, dz = x-xsun, y-ysun, z-zsun
    dr = np.sqrt(dx*dx+dy*dy+dz*dz)

    return ((ax-axsun)*dx + (ay-aysun)*dy + (az-azsun)*dz)/(np.maximum(dr,1.e-10))

def alos_from_pulsar(d, b, l, mus, alos_gr, parameters) : 
    ## pulsar positions --
    x = d * np.cos(b*np.pi/180.) * np.cos(l*np.pi/180.) *kpctocm + xsun
    y = d * np.cos(b*np.pi/180.) * np.sin(l*np.pi/180.) * kpctocm + ysun
    z = d * np.sin(b*np.pi/180.) * kpctocm + zsun

    al = alos(x,y,z,parameters)

    # include sloskey effect mu^2 *d /c
    al_sl = mus*mus*d*kpctocm/c

    return al+al_sl + alos_gr

def string_to_value_and_error( string, has_exponent=False) :
    import re
    from decimal import Decimal
    substring = r"(-*\d+\.\d+)\((\d+)\)"
    if( has_exponent) :
        substring = r"(-*\d+\.\d+)\((\d+)\)e(-\d+)$"

    value = re.sub(substring,r"\1", string)
    error = re.sub(substring,r"\2", string)
    err_exp = Decimal(value).as_tuple().exponent
    if( has_exponent) :
        exp = re.sub(substring, r"\3", string)
        exponent = 10.**float(exp)
        return float(value)*exponent, float(error)*10**err_exp*exponent
    return float(value), float(error)*10**err_exp

def initialize_theta( frac_random=1) :
    global pulsar_data 
    number_pulsars = pulsar_data["number pulsars"]
    distances = pulsar_data["distance"]
    distance_err = pulsar_data["distance error"]
    alos_gr = pulsar_data["alos_gr"]
    alos_gr_err = pulsar_data["alos_gr_err"]
    mus = pulsar_data["mu"]
    mu_err = pulsar_data["mu error"]
    theta = np.zeros(number_parameters+3*number_pulsars)
    if MODEL == QUILLEN : 
        lgalpha1 = lgalpha1_0 + 0.1*np.random.randn(1)[0]*frac_random
        lgalpha2 = lgalpha2_0 - 0.1*np.random.randn(1)[0]*frac_random
        theta[0] = lgalpha1
        if(number_parameters > 1) :
            theta[1] = lgalpha2
    elif MODEL == QUILLENBETA : 
        lgalpha1 = lgalpha1_0 + 0.1*np.random.randn(1)[0]*frac_random
        beta = beta0 - 0.1*np.random.randn(1)[0]*frac_random
        theta[0] = lgalpha1
        if(number_parameters > 1):
            theta[1] = beta
    elif MODEL == EXPONENTIAL or MODEL == GAUSSIAN or MODEL == SECH2: 
        lgrho0 = lgrho_midplane + 0.1*np.random.randn(1)[0]*frac_random
        lgz0 = lgscale_height + 0.1*np.random.randn(1)[0]*frac_random
        theta[0] = lgrho0
        theta[1] = lgz0
    elif MODEL == HERNQUIST :
        lgMh = lgMh_0 + 0.1*np.random.randn(1)[0]*frac_random 
        lga = lga_0 + 0.01*np.random.randn(1)[0]*frac_random
        theta[0] = lgMh
        theta[1] = lga
    elif MODEL == HALODISK:
        lgMh = lgMh_0 + 0.1*np.random.randn(1)[0]*frac_random
        lga = lga_0 + 0.01*np.random.randn(1)[0]*frac_random
        lgrho0 = lgrho_midplane + 0.1*np.random.randn(1)[0]*frac_random
        lgz0 = lgscale_height + 0.1*np.random.randn(1)[0]*frac_random
        theta[0] = lgMh
        theta[1] = lga
        theta[2] = lgrho0
        theta[3] = lgz0 
    elif MODEL == MWpot and number_parameters > 0:
        #theta[0] = alpha_mw2014 + alpha_mw2014_range*np.random.randn(1)[0]*frac_random
        #theta[1] = rc_mw2014 + rc_mw2014_range*np.random.randn(1)[0]*frac_random
        theta[0] = lg_a0_mw2014 + lg_a0_mw2014_range*np.random.randn(1)[0]*frac_random
        theta[1] = lg_b0_mw2014 + lg_b0_mw2014_range*np.random.randn(1)[0]*frac_random
    elif MODEL == LOCAL : 
        theta[0] = dadr0 + 0.1*dadr_range*np.random.randn(1)[0]*frac_random
        theta[1] = dadphi0 + 0.1*dadphi_range*np.random.randn(1)[0]*frac_random
        theta[2] = lgdadz0 + 0.1*lgdadz_range*np.random.randn(1)[0]*frac_random
    elif MODEL == POWER_LAW : 
        theta[0] = lgalpha1_0 + 0.1*np.random.randn(1)[0]*frac_random
        #theta[1] = eta0 + 0.1*eta_range*np.random.randn(1)[0]*frac_random
        theta[1] = zmid0 + 0.1*zmid_range*np.random.randn(1)[0]*frac_random

#    if(number_parameters > 2) :   ## CHECK THIS
# I DON'T THINK THIS IS NEEDED
#        theta[2] = Vlsr0 + Vlsr_err*np.random.randn(1)[0]*frac_random
    theta[number_parameters:number_pulsars+number_parameters] = distances + distance_err*np.random.randn(number_pulsars)*frac_random
    theta[number_pulsars+number_parameters:2*number_pulsars+number_parameters] = mus + mu_err*np.random.randn(number_pulsars)*frac_random
    theta[2*number_pulsars+number_parameters:] = alos_gr + alos_gr_err*np.random.randn(number_pulsars)*frac_random

    return theta

def unpack_theta( theta, number_pulsars) :
    parameters = None
    if( number_parameters > 0) :
        parameters = theta[0:number_parameters]
    
    distances = theta[number_parameters:number_pulsars+number_parameters]
    mus = theta[number_parameters+number_pulsars:number_parameters+2*number_pulsars]
    alos_gr = theta[number_parameters+2*number_pulsars:]
    return parameters, distances, mus, alos_gr

def model_and_data( theta) : 
    global pulsar_data
    number_pulsars = pulsar_data["number pulsars"]
    alos_model = None
    if( not theta is None) :
        parameters, distances, mus, alos_gr = unpack_theta(theta, number_pulsars)
        b = pulsar_data["latitude"]
        l = pulsar_data["longitude"]
        alos_model = alos_from_pulsar( distances, b, l, mus, alos_gr, parameters)
    alos_obs = pulsar_data["alos"]
    alos_err = pulsar_data["alos error"]

    return alos_model, alos_obs, alos_err

def log_likelihood( theta, return_chisq = False) :
    alos_model, alos_obs, alos_err = model_and_data( theta)
    if( not return_chisq) :
        ll = 0.5*np.sum( -((alos_obs - alos_model)/alos_err)**2 - np.log(2*np.pi*alos_err*alos_err))
        ll = 0.5*np.sum( -((alos_obs - alos_model)/alos_err)**2)# - np.log(2*np.pi*alos_err*alos_err))
        return ll
    else :
        return np.sum(((alos_obs - alos_model)/alos_err)**2), ((alos_obs - alos_model)/alos_err)**2

def log_prior( theta) :
    global pulsar_data 
    number_pulsars = pulsar_data["number pulsars"]
    parameters, distances, mus, alos_gr = unpack_theta(theta, number_pulsars)

    distance_err = pulsar_data["distance error"]
    mu_err = pulsar_data["mu error"]
    distance_arr = pulsar_data["distance"]
    mu_arr = pulsar_data["mu"]
    alos_err = pulsar_data["alos error"]
    alos_gr_err = pulsar_data["alos_gr_err"]
    alos_gr_arr = pulsar_data["alos_gr"]

    # define the range in alpha1, alpha2
    lp = 0
#    print("Here")
    if MODEL == QUILLEN : 
        lgalpha1 = parameters[0]
        if( np.abs(lgalpha1 - lgalpha1_0) > p1_range)  :
            return -np.inf
        if(number_parameters > 1) :     
            lgalpha2 = parameters[1]
            if( np.abs(lgalpha2 - lgalpha2_0) > p2_range) :
                return -np.inf

    elif MODEL == QUILLENBETA:
        lgalpha1 = parameters[0]
        if(np.abs(lgalpha1 - lgalpha1_0) > p1_range)  :
            return -np.inf
        if(number_parameters > 1):
            beta = parameters[1]
            if( np.abs(beta - beta0) > brange) :
                return -np.inf
    elif MODEL == EXPONENTIAL or MODEL == GAUSSIAN or MODEL == SECH2:
        lgrho0, lgz0 = parameters[0:2] 
        if( np.abs(lgrho0 - lgrho_midplane) > p1_range)  :
            return -np.inf
        if(number_parameters > 1 and np.abs(lgz0 - lgscale_height) > p2_range) :
            return -np.inf
    elif MODEL == HERNQUIST:
        lgMh, lga = parameters[0:2]
        if (np.abs(lgMh - lgMh_0) > Mhrange) :
            return -np.inf
        #lp+=-0.5*((lga-lga_0)/p2_range)**2
        if (np.abs(lga - lga_0) > p2_range):
            return -np.inf
    elif MODEL == HALODISK:
        lgMh, lga, lgrho0, lgz0 = parameters[0:4]
        if (np.abs(lgMh - lgMh_0) > p1_range) :
            return -np.inf
        if (np.abs(lga - lga_0) > p2_range):
            return -np.inf
        if( np.abs(lgrho0 - lgrho_midplane) > p1_range)  :
            return -np.inf
        if(number_parameters > 1 and np.abs(lgz0 - lgscale_height) > p2_range) :
            return -np.inf
    elif MODEL == MWpot : 
        if( number_parameters > 0 ) : 
            lga, lgb = parameters[0:2]
            #lp += -0.5*((alpha-alpha_mw2014)/alpha_mw2014_range)**2
            #lp += -0.5*((rc-rc_mw2014)/rc_mw2014_range)**2
            lp += -0.5*((lga-lg_a0_mw2014)/lg_a0_mw2014_range)**2
            lp += -0.5*((lgb-lg_b0_mw2014)/lg_b0_mw2014_range)**2
    elif MODEL == LOCAL : 
        dadr, dadphi, lgdadz = parameters[0:3]
        if( np.abs(dadr - dadr0) > dadr_range ) : 
            return -np.inf
        if( np.abs(dadphi-dadphi0 ) > dadphi_range) : 
            return -np.inf
        if( np.abs(lgdadz - lgdadz0) > lgdadz_range) :
            return -np.inf
    elif MODEL == POWER_LAW : 
        lgalpha1, zmid = parameters[0:2]
        if( np.abs(lgalpha1 - lgalpha1_0) > p1_range)  :
            return -np.inf
        #if( np.abs(eta - eta0) > eta_range) :
        #    return -np.inf
        if( np.abs(zmid - zmid0) > zmid_range) :
            return -np.inf

    #lp += -0.5*(((Vlsr - Vlsr0)/Vlsr_err)**2 + math.log(2*math.pi*Vlsr_err**2))
    lp += 0.5*np.sum(-((distances-distance_arr)/distance_err)**2)# - np.log(2*np.pi*distance_err**2))
    lp += 0.5*np.sum(- ((mus-mu_arr)/mu_err)**2)# - np.log(2*np.pi*mu_err**2))
    lp += 0.5*np.sum(- ((alos_gr-alos_gr_arr)/alos_gr_err)**2)# - np.log(2*np.pi*mu_err**2))

    return lp

def log_probability(theta) :
    global pulsar_data
    lp = log_prior( theta)
    if not np.isfinite( lp) : 
        return -np.inf
    return lp + log_likelihood( theta)

def read_pulsar_data() :
    global pulsar_data 
    import pandas as pd

    # import the GR corrections
    gr_dataframe = pd.read_excel("PBDOT_GR.xls")
    gr_names = gr_dataframe["Pulsar Name"]
    gr_pbdot = gr_dataframe["PBDOT_GR (s/s)"]

    # import the data and decode
    dataframe = pd.read_excel("PBDOT.xls") 
    distances = dataframe['Parallax Distance (kpc)'] 
    names = dataframe["Pulsar Name"]
    datasets = dataframe["Dataset"]
    mus = dataframe["Proper Motion (mu, mas/yr)"]
    pbs = dataframe["PB (d)"]
    pbdot_obs = dataframe["PBDOT_obs (s/s)"]
    longitude = dataframe["Galactic Longitude (l, deg)"]
    latitude = dataframe["Galactic Latitude (b, deg)"]
    other_distances = dataframe["Other Distance (kpc)"]

    name_arr = []
    dataset_arr = []
    distance_arr = []
    distance_err = []
    latitude_arr = []
    longitude_arr = []
    mu_arr = []
    mu_err = []

    alos_gr_arr = []
    alos_gr_err = []
    alos_arr = []
    alos_err = []

    for name, dataset, distance_str, other_distance_str, mu_str, pb_str, pbdot_str, lat_str, long_str, pulsar_no in \
            zip(names, datasets, distances, other_distances, mus, pbs, pbdot_obs, latitude, longitude, range(names.size)):
        if( (unique_name and name in name_arr) or (not excluded_pulsars is None and name in excluded_pulsars)) : 
            continue
        if( not pulsars_number is None and not pulsar_no in pulsars_number) :
            continue
        if( not best_pulsars is None and not pulsar_no in best_pulsars) :
            continue
        name_arr.append(name) 
        dataset_arr.append(dataset)
        d, derr = string_to_value_and_error(distance_str)

        if(pulsar_no in use_other_distance) : 
            d, derr = string_to_value_and_error(distance_str)

        pbdot_gr = TINY_GR
        pbdot_gr_err = TINY_GR
        if(name in gr_names.values) : 
            pbdot_gr_str = gr_pbdot.values[gr_names.values == name][0]
            pbdot_gr, pbdot_gr_err = string_to_value_and_error( pbdot_gr_str, has_exponent=True)

        mu, muerr = string_to_value_and_error( mu_str)
        mu *= mas_per_year_to_as_per_sec
        muerr *= mas_per_year_to_as_per_sec
        pbdot, pbdoterr = string_to_value_and_error( pbdot_str, has_exponent=True) 
        pb, pberr = string_to_value_and_error( pb_str)
        pb *= day_to_sec

        distance_arr.append(d)
        distance_err.append( derr)
        mu_arr.append( mu)
        mu_err.append( muerr)
        alos_arr.append(pbdot/pb*c)
        alos_err.append(pbdoterr/pb*c)
        latitude_arr.append(float(lat_str))
        longitude_arr.append(float(long_str))
        alos_gr_arr.append(float(pbdot_gr/pb*c))
        alos_gr_err.append(float(pbdot_gr_err/pb*c))
        print(name, dataset)

    longitude_arr = np.array(longitude_arr)
    latitude_arr = np.array(latitude_arr)
    distance_arr = np.array(distance_arr)
    distance_err = np.array(distance_err)
    mu_arr = np.array(mu_arr)
    mu_err = np.array(mu_err)
    alos_arr = np.array(alos_arr)
    alos_err = np.array(alos_err)
    alos_gr_arr = np.array(alos_gr_arr)
    alos_gr_err = np.array(alos_gr_err)

    pulsar_data = {"name" : name_arr, "dataset": dataset_arr, "longitude": longitude_arr, "latitude" : latitude_arr, \
                    "distance" : distance_arr, "distance error" : distance_err, "mu" : mu_arr, "mu error" : mu_err, \
                    "alos_gr" : alos_gr_arr, "alos_gr_err" : alos_gr_err, \
                    "alos" : alos_arr, "alos error" : alos_err, "number pulsars" : len(name_arr)}

def run_samples( sampler, pos, iterations, min_steps=1000, tau_multipler=100) : 
    import time
    current_iteration = 0
    stop = False
    autocorr = []
    old_tau = np.inf
    while not stop : 
        nstep = min(min_steps, iterations-current_iteration) 
        start = time.time()
        sampler.run_mcmc(pos, nstep, progress=True)
        end = time.time()
        pos = None # run from current point
        current_iteration += nstep
        tau = sampler.get_autocorr_time(tol=0)
        autocorr.append(np.mean(tau))
        
        # Check convergence
        criteria1 = np.all(tau * tau_multipler < sampler.iteration)
        criteria2 = np.all(np.abs(old_tau - tau) / tau < 0.01)
        converged = criteria1 and criteria2

        if(converged or current_iteration >= iterations) : 
#        if (current_iteration >= iterations):  ## to avoid this
            stop = True
        print("step: {0:07d}, mean tau: {1:5.2e}, conv. crit: {3}, steps to conv: {4:5.2e}, it/s: {5:5.2e}".format(current_iteration, np.mean(tau), criteria1, criteria2, np.max(tau_multipler*tau), nstep/(end-start)))

        old_tau = tau
    
    if MODEL == HERNQUIST:
        labels = ["Mh","a"]
    elif MODEL == QUILLEN:
        labels = ["alpha1"]
        if number_parameters > 1:
            labels = ["alpha1","alpha2"]
    elif MODEL == QUILLENBETA:
        if number_parameters > 1:
            labels = ["alpha1","beta"]
    elif MODEL == EXPONENTIAL or GAUSSIAN:
        labels = ["rho0","z0"]
    elif MODEL == HALODISK: 
        labels == ["Mh","a","rho0","z0"]

    # fig, axes = pl.subplots(number_parameters, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()    

    # if not (chainplot == 0):
    #     for i in range(number_parameters):
    #         ax = axes[i]
    #         ax.plot(samples[:, :, i], "k", alpha=0.3)
    #         ax.set_xlim(0, len(samples))
    #         ax.set_ylabel(labels[i])
    #         ax.yaxis.set_label_coords(-0.1, 0.5)

    #         axes[-1].set_xlabel("step number")
    #         pl.savefig("chain.png")
    # pl.clf()    

    return autocorr


def run_mcmc() :
    import time
    itheta = initialize_theta( frac_random=0.)

    nwalkers = 100
    ndim = itheta.size

    pos = np.zeros( [nwalkers,ndim])

    for i in range(nwalkers) : 
        pos[i,:] = initialize_theta()

    import multiprocessing

    backend = None
    if( not filename is None) : 
        if os.path.exists(filename):
            os.remove(filename)

        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)

    sampler = None
    current_iteration = 0
    start = time.time()
    if multithread : 
        with multiprocessing.Pool(processes=processes) as pool : 
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool)
            run_samples(sampler, pos, iterations)
    else :
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend)
        run_samples(sampler, pos, iterations)
        #sampler.run_mcmc(pos, iterations, progress=True)
    end = time.time()

    print("Iterations {0} took {1:.1f} seconds; average it/s: {2:.2f}".format(iterations, end-start, iterations/(end-start)))
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=discard, flat=True, thin=thin)
    
    return flat_samples, flat_log_prob

def read_samples(filename=filename):
    reader = emcee.backends.HDFBackend(filename)
    flat_samples = reader.get_chain(discard=discard, flat=True, thin=thin)
    flat_log_prob = reader.get_log_prob(discard=discard, flat=True, thin=thin)
    return flat_samples, flat_log_prob

def get_best_fit( flat_samples, flat_log_prob, show_chi_sq_arr = True) : 
    itheta = initialize_theta( frac_random=0.)
    best_fit_theta = np.zeros(itheta.shape)
    print(itheta.shape, flat_samples.shape)
    for i in range(best_fit_theta.size) : 
       mcmc = np.percentile(flat_samples[:, i], [50])
       best_fit_theta[i] = mcmc[0]
 
    chi_sq, chi_sq_arr = log_likelihood( best_fit_theta, return_chisq =True)
    print("best fit chisq = ", chi_sq)
    if( show_chi_sq_arr) :
        print("Chi square array = ", chi_sq_arr)
    return best_fit_theta, chi_sq


def plot_model(theta, include_labels=True, obs=True,model=True, model_label=None, logy=False) : 
    global pulsar_data

    x = range(len(pulsar_data["name"]))
    alos_model, alos_obs, alos_err = model_and_data(theta)

    if( obs) : 
        if( logy) :
            pl.errorbar(x,np.abs(alos_obs), yerr=alos_err, fmt="o",alpha=0.25,c='black',label=r"$a_{\rm los, obs}$")
        else : 
            pl.errorbar(x,alos_obs, yerr=alos_err, fmt="o",alpha=0.25,c='black',label=r"$a_{\rm los, obs}$")

    if( model) : 

        if( model_label is None) :
            model_label = r"$a_{\rm los, mod}$"
        if( logy):
            pl.scatter(x,np.abs(alos_model), s=8, alpha=1,label=model_label)
        else : 
            pl.scatter(x,alos_model, s=8, alpha=1,label=model_label)

    if( include_labels) : 
        names = np.array(pulsar_data["name"])
        pl.ylabel(r"$|a_{\rm los}|\,[{\rm cm\,s}^{-2}]$")
        pl.xticks(range(len(names)), labels=names, rotation="vertical")
        if(logy) : 
            pl.ylim(1e-11,1e-6)
            pl.yscale('log')
        #pl.xscale('log')
        pl.legend(loc="best")

def make_corner_plot(flat_samples, flat_log_prob) :
    global pulsar_data
    import corner
    
    best_fit_theta, _ = get_best_fit( flat_samples, flat_log_prob)    

    if number_parameters > 0 : 

        pl.clf()
        labels = []
        for i in range(number_parameters) : 
            labels.append("p{0} = ".format(i))
        #labels = [r"$\log(Mh)$",r"$\log(a)$", r"$\log(rho0)$", r"$\log(z0)$", r"$V_{\rm lsr}$"]
        if MODEL == QUILLEN:
            labels = [r"$\alpha1$",r"$\alpha2$"]
        elif MODEL == QUILLENBETA:
            labels = [r"$\alpha1$",r"$beta$"]
        if MODEL == EXPONENTIAL or MODEL == GAUSSIAN or MODEL == SECH2: 
            labels = [r"$\log(\rho_0/1\,M_{\odot}\,{\rm pc}^{-3})$",r"$\log(z_0/1\,{\rm pc})$", r"$V_{\rm lsr}$"]
            flat_samples[:,0] -= math.log10(2e33/pc**3)
            flat_samples[:,1] -= math.log10(pc)
        fig = corner.corner( flat_samples[:,0:number_parameters], labels=labels[0:number_parameters],truths=best_fit_theta[0:number_parameters])
        pl.savefig("corner.png")

        for i in range(number_parameters):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            print( "{0} = {1} {2} {3}".format(labels[i], mcmc[1], q[0], q[1]))

    pl.clf()
    plot_model(best_fit_theta)
    #print(alos_model)
    #print(alos_obs)
    #print(alos_err)

    # inds = np.random.randint(len(flat_samples), size=1000)

    # for ind in inds:
    #     theta = flat_samples[ind]
    #     alos_model, alos_obs, alos_err = model_and_data(best_fit_theta)
    #     pl.scatter(range(alos_model.size), alos_model, c='red', s=2, alpha=0.5)

    pl.savefig("test.pdf")

def run_model() :
    initialize_model( MODEL)
    read_pulsar_data()
    flat_samples = None
    if( args.load_previous) :
        flat_samples, flat_log_prob = read_samples(filename)
    else : 
        flat_samples, flat_log_prob = run_mcmc()

    make_corner_plot( flat_samples, flat_log_prob)

def run_compilation() : 
    models = [QUILLEN, QUILLENBETA, EXPONENTIAL, MWpot, LOCAL, Hernquistfix, HALODISK]
    rootdir = "../../../../Code/test-PTA"
    files = ["quillen.h5", "quillen_beta.h5", "exp.h5", "mw2014.h5", "local.h5", "hernquist_fixed.h5", "halodisk.h5"]
    labels = ["Quillen", r"Quillen + $\beta$", r"$\exp(-|z|/h_z)$", "MWP2014", "local", "Hernquist", "Hernquist+disk"]

    #models = [QUILLEN, LOCAL]
    #files = ["quillen.h5", "local.h5"]
    #labels = ["Quillen", "local"]
    read_pulsar_data()

    plot_model(None, include_labels=False, obs=True, model=False)

    for model, f, label in zip(models, files, labels) :
        initialize_model(model=model)
        fname = "{0}/{1}".format(rootdir, f)
        flat_samples, flat_log_prob = read_samples(fname)
        
        theta, chisq = get_best_fit( flat_samples, flat_log_prob)
        
        for i in range(number_parameters):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            print( "{0} = {1} {2} {3}".format("p{0}".format(i), mcmc[1], q[0], q[1]))
 
        plot_model(theta, include_labels=False, obs=False, model=True, model_label=r"{0}: $\chi^2 = {1:.1f}$".format(label, chisq))

    plot_model(None, include_labels=True, obs=False, model=False)
    pl.tight_layout()
    pl.savefig("test.pdf")


import argparse
parser = argparse.ArgumentParser(description='Run calibration')
parser.add_argument('-o', help="hdf5 file to store/load mcmc chain")
parser.add_argument('--load_previous', action='store_true',
                    default=False,
                    help="load MCMC from file")
parser.add_argument('--multi', action="store_true",
                    default=None,
                    help="use multiprocessing")
parser.add_argument('--num_procs', type=int, default=0, help="set number of processes to use for multiprocessing")
parser.add_argument('--model', type=int, default=DEFAULT_MODEL, help="set model to be run")
parser.add_argument('--compilation', action="store_true", default=False, help="run the compilation")
args = parser.parse_args()

if not args.o is None :
    filename = args.o

multithread = args.multi
MODEL = args.model

if( args.num_procs > 0) : 
    multithread = True
    processes = args.num_procs

if( args.compilation) : 
    run_compilation()
else :
    run_model()
