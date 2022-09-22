import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

def _mean_anomaly(nu, e):
    integrand = lambda x: 1./(1.+e*np.cos(x))**2
    integral, _ = quad(integrand, 0., nu)
    result = ((1.-e**2)**1.5)*integral
    return result

class orbit():

    def __init__(self, Mass=1., a=1., e=0., omega=0., nu=None, m=None, G=4.*np.pi**2):

        # Gravitational constant
        self.G = G

        # Core variables
        self.Mass = Mass
        self.a = a
        self.e = e
        self.omega = np.deg2rad(omega)

        # Anomaly angles
        if (nu is None) and (m is None):
            raise ValueError('You must specify either mean anomaly or true anomaly.')
        elif nu is None:
            self.m = np.deg2rad(m)
            self.nu = self.true_anomaly()
        elif m is None:
            self.nu = np.deg2rad(nu)
            self.m = self.mean_anomaly()
        else:
            raise ValueError('Cannot specify both mean anomaly and true anomaly')

        # Calculations
        self.P = self.Period()
        self.theta = self.omega+self.nu
        
        # Vectorisations
        self.time = np.vectorize(self._time)
        self.angle = np.vectorize(self._angle)

    def __str__(self):
        print('== Constants ==')
        print('Gravitational constant:', self.G)
        print('== Parameters ==')
        print('Central mass:', self.Mass)
        print('Semi-major axis:', self.a)
        print('Eccentricity:', self.e)
        print('Argument of periapsis [deg]:', np.rad2deg(self.omega))
        print('== Anomalies ==')
        print('True anomaly [deg]:', np.rad2deg(self.nu))
        print('Mean anomaly [deg]:', np.rad2deg(self.m))
        print('== Derived Parameters ==')
        print('Period:', self.P)
        print('Theta [deg]:', np.rad2deg(self.theta))
        return ''


    def mean_anomaly(self, approximate=False):
        '''
        Compute the mean anomaly corresponding to the true anomaly via integration
        '''
        nu = self.nu; e = self.e
        if approximate: # Taylor expansion for low e
            m = nu-2.*e*np.sin(nu)
        else: # Integral for full result
            m = _mean_anomaly(nu, e)
        return m


    def true_anomaly(self, approximate=False):
        '''
        Compute the true anomaly from the mean anomaly using root finding
        '''
        m = self.m; e = self.e
        if approximate:
            nu = m+2.*e*np.sin(m) # Approximate numerical inversion for small e
        else:
            f = lambda x: _mean_anomaly(x, e)-m
            #nu_guess = m # This is probably a pretty good first guess
            nu_guess = self.true_anomaly(approximate=True) # Use approximate (e<<1) solution as first guess
            nu = fsolve(f, nu_guess)[0] # Initial guess that nu = M
        return nu


    def semi_major_axis(self):
        '''
        Get the orbital semi-major axis as a function of planet period and star mass
        Kepler's third law with units such that time: [years]; mass [Solar]; distance [au]
        '''
        M = self.Mass; P = self.P; G = self.G
        return ((G*M*P**2)/(4.*np.pi**2))**(1./3.)


    def Period(self):
        '''
        Get the orbital period as a function of planet period and star mass
        Kepler's third law with units such that time: [years]; mass [Solar]; distance [au]
        '''
        G = self.G; M = self.Mass; a = self.a
        return np.sqrt(4.*np.pi**2*a**3/(G*M))


    def coordinates_theta(self, theta):
        '''
        Convert orbital parameters into x, y
        '''
        a = self.a; e = self.e; omega=self.omega
        fac = a*(1.-e**2)/(1.+e*np.cos(theta-omega))
        return fac*np.cos(theta), fac*np.sin(theta)


    def _time(self, theta):
        '''
        Calculate the time along the orbit as a function of angle
        '''
        P = self.P; e = self.e; omega = self.omega; nu=self.nu
        fac = ((1.-e**2)**1.5)/(2.*np.pi)
        def integrand(theta, e, omega):
            return 1./(1.+e*np.cos(theta-omega))**2
        integral, _ = quad(lambda x: integrand(x, e, omega), a=omega+nu, b=theta)
        time = P*fac*integral
        return time


    def _angle(self, t):
        '''
        Calculate the orbital angle as a function of time
        '''
        P = self.P
        dt = t%P
        theta_init = 2.*np.pi*dt/P
        theta = fsolve(lambda x: self.time(x)-dt, theta_init)
        return theta[0]


    def coordinates(self, t):
        '''
        Get the orbital x, y coordinates as a function of time
        '''
        thetas = self.angle(t)
        x, y = self.coordinates_theta(thetas)
        return x, y