import math
import numpy
import constants
import scipy.integrate as ode


class Orbits :
    def __init__( self) :
        self.particles = []
        self.perturbers = []
        self.currentTime = 0.        

    def add_perturber(self,perturber) :
        self.perturbers.append(perturber) 

    @staticmethod
    def derivatives( t, y, orbits) :    
        n_particles = len(orbits.particles)
        n_perturbers = len(orbits.perturbers)
        tot_particles = n_perturbers+n_particles

        z = y.reshape(tot_particles,2,3)
        
        pert_positions = z[0:n_perturbers, 0, :]
        pert_velocities = z[0:n_perturbers, 1, :]

        positions = z[:,0,:]
        velocities = z[:,1,:]
        
        derivs = numpy.zeros([tot_particles,2,3])

        #    update perturber positions first
        for j in range(n_perturbers) :
            orbits.perturbers[j].update(pert_positions[j,:], pert_velocities[j,:])
            #print "orbit positions: " + str(orbits.perturbers[j].pos)
    
        for i in range(tot_particles) :

            pos = positions[i,:]
            vel = velocities[i,:]
            r = math.sqrt(numpy.dot(pos,pos))
            dposdt = vel
            dveldt = numpy.zeros(3)

            mass = -1
            if( i < n_perturbers) :
                mass = orbits.perturbers[i].profile.get_mass()

            for j in range(n_perturbers) :
                if( i != j) : # no self force
                    dveldt += orbits.perturbers[j].get_acceleration( pos, vel, mass)

            derivs[i,0,:] = dposdt
            derivs[i,1,:] = dveldt

        derivs = derivs.reshape(derivs.size)
        return derivs

    def set_current_time( self, time) :
        self.currentTime = time
    
    def evolve( self, dt) :
        y0 = []
        integrator = ode.ode(Orbits.derivatives).set_integrator('dop853', rtol=1e-4, dfactor=2.0, first_step=0.01*dt, max_step=2.0*constants.Myr, nsteps=1000).set_f_params(self)
        
        for particle in self.perturbers :
            y0.append([particle.pos, particle.vel])
 
        for particle in self.particles :
            y0.append([particle.pos, particle.vel])
           
        y0 = numpy.array(y0)
        integrator.set_initial_value( y0.reshape(y0.size), self.currentTime)
        #print 'time = ' + str(integrator.t) + " " + str(dt)
        integrator.integrate( integrator.t + dt)

        self.currentTime = integrator.t
    
        n_particles = len(self.particles)
        n_perturbers = len(self.perturbers)
        z = integrator.y.reshape(n_particles+n_perturbers,2,3)

        # update perturbers
        for i in range(n_perturbers) :
            pos = z[i,0,:]
            vel = z[i,1,:]
            self.perturbers[i].update( pos, vel)

        for i in range( n_particles) :
            pos = z[i+n_perturbers,0,:]
            vel = z[i+n_perturbers,1,:]
            self.particles[i].update( pos, vel)

        return

