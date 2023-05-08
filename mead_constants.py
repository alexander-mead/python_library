# Try https://docs.scipy.org/doc/scipy/reference/constants.html

# Third-party imports
from scipy import constants as const

# Mathematical constants
pi = const.pi # Lovely pi

# Physical constants
G = const.gravitational_constant # Newton constant [m^3 kg^-1 s^-2]
h = const.Planck                 # Planck constant [kg m^2 s^-1]
c = const.speed_of_light         # Speed of light [m s^-1]
kB = const.Boltzmann             # Boltzmann constant [m^2 kg s^-2 K^-1]
mp = const.proton_mass           # Proton mass [kg]
a_Wein = 5.879e10                # Wein constant [Hz/K]

# Units
Jy = 1e-26                   # Jansky [W m^-2 Hz^-1]
au = const.astronomical_unit # au [m] ~1.496e11 m
eV_mass = const.eV/c**2      # eV [kg] ~1.783 kg

# Time
seconds_in_day = 60*60*24 # [s]

# Astronomy
Sun_radius = 6.96340e8    # Radius of the Earth [m]
Sun_mass = 1.9884e30      # Mass of the Sun [kg]
Earth_mass = 5.9722e24    # Mass of the Earth [kg]
Jupiter_mass = 1.898e27   # Mass of Jupiter [kg]
year = const.Julian_year  # year [s] ~3.156e7 s
pc = const.parsec         # parsec [m] ~3.0857e16 m
Mpc = 1e6*pc              # Megaparsec [m]

# Cosmology
G_cos = G*Sun_mass/(Mpc*1e3**2)       # Gravitational constant [(Msun/h)^-1 (km/s)^2 (Mpc/h)] ~4.301e-9 (1e3**2 m -> km)
H0_cos = 100.                         # H0 in h [km/s/Mpc] 100 km/s/Mpc
H0 = H0_cos*1e3/Mpc                   # H0 [s^-1] ~3.241 s^-1
Hdist_cos = c/(H0_cos*1e3)            # c/H0 [Mpc/h] ~2998 Mpc/h
Htime_cos = 1./(H0*year*1e9)          # 1/H0 [Gyr/h] ~9.778 Gyr/h
rhoc_cos = 3.*H0_cos**2/(8.*pi*G_cos) # Critical density [(Msun/h) (Mpc/h)^-3] ~2.775e11 (Msun/h)/(Mpc/h)^3
nuconst = 93.1                        # Neutrino mass required to close universe [eV]
