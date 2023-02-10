# Standard imports
import numpy as np

# Project imports (come from pylians)
import readgadget
import readfof

# For use with Quijote

def read_gadget(snapshot, ptype=[1]):
    
    # Read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  # Mpc/h
    Nall     = header.nall         # Total number of particles
    Masses   = header.massarr*1e10 # Masses of the particles in Msun/h
    Omega_m  = header.omega_m      # Value of Omega_m
    Omega_l  = header.omega_l      # Value of Omega_l
    h        = header.hubble       # Value of h
    redshift = header.redshift     # Redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l) # Value of H(z) in km/s/(Mpc/h)
    
    # Print information
    print('Gadget particles')
    print('Box size [Mpc/h]:', BoxSize)
    print('Number of particles:', Nall[1], Nall[1]**(1./3.))
    print('Masses [Msun/h]', Masses[1])
    print('Omega_m:', Omega_m)
    print('Omega_L:', Omega_l)
    print('h:', h)
    print('Redshift:', redshift)
    print('Hubble [h km/s/Mpc]:', Hubble)

    # Read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 # Positions in [Mpc/h]
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     # Peculiar velocities in [km/s]
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   # IDs starting from 0
    
    # Print more information
    for i, name in zip([0, 1, 2], ['x', 'y', 'z']):
        print('Min/max position [Mpc/h]:', name, pos[:, i].min(), pos[:, i].max())
    print()
    
    return pos, vel, ids

def read_fof(snapdir, snapnum):
    
    # Determine the redshift of the catalogue
    z_dict = {4:0., 3:0.5, 2:1., 1:2., 0:3.}
    redshift = z_dict[snapnum]
    
    # Print information
    print('FoF haloes')
    print('Redshift:', redshift)

    # Read the halo catalogue
    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)

    # Get the properties of the haloes
    pos = FoF.GroupPos/1e3            # Halo positions [Mpc/h]
    vel = FoF.GroupVel*(1.0+redshift) # Halo peculiar velocities [km/s]
    mass  = FoF.GroupMass*1e10          # Halo masses [Msun/h]
    Npart = FoF.GroupLen                # Number of CDM particles in the halo
    
    print('Number of haloes:', len(mass))
    # Print more information
    for i, name in zip([0, 1, 2], ['x', 'y', 'z']):
        print('Min/max position [Mpc/h]:', name, pos[:, i].min(), pos[:, i].max())
    print()
    print()
    
    return pos, vel, mass, Npart