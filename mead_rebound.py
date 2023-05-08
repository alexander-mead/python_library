# Third-party imports
import numpy as np

def specific_angular_momentum_vector(particle):
    '''
    Calculate the specific angular momentum vector of a particle instance
    Calcuated as 'r x v'
    '''
    r = np.array([particle.x, particle.y, particle.z])
    v = np.array([particle.vx, particle.vy, particle.vz])
    return np.cross(r, v)

def semi_major_axis(P, M_star, sim):
    '''
    Use Kepler's third law to calculate semi-major axis from orbital period
    '''
    f1 = M_star*sim.G*P**2
    f2 = 4.*np.pi**2
    return (f1/f2)**(1./3.)

def rotate_transit_particles(sim):
    '''
    Rotate the simulation particles appropriate for visualising transiting particles
    '''
    for particle in sim.particles:
        x = particle.x; y = particle.y; z = particle.z
        vx = particle.vx; vy = particle.vy; vz = particle.vz
        particle.x = x; particle.y = z; particle.z = y
        particle.vx = vx; particle.vy = vz; particle.vz = vy
        #particle.x = z; particle.y = y; particle.z = x
        #particle.vx = vz; particle.vy = vy; particle.vz = vx

def reverse_motion(particle):
    '''
    Reverse the direction of motion of a particle
    '''
    particle.vx = -particle.vx
    particle.vy = -particle.vy
    particle.vz = -particle.vz