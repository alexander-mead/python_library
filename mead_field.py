# Third-party imports
import numpy as np


def make_Gaussian_random_field_2D(mean_value: float, power_spectrum: np.ndarray,
                                  map_size: int, mesh_cells: int, *args,
                                  periodic=True) -> np.ndarray:

    # mean_value: mean value for the field
    # power_spectrum: P(k, *args) power spectrum for the field [(length units)^2]
    # map_size: side length for the map [length units]
    # mesh_cells: number of mesh cells for the field
    # periodic: should the map be considered to be periodic or not?
    # *args: extra arguments for power spectrum

    # TODO: Enforce Hermitian condition
    # TODO: Use real-to-real FFT
    # TODO: Generalise to nD

    # Parameters
    pad_fraction = 2  # Amount to increase size of array if non-periodic

    if periodic:
        tran_cells = mesh_cells
        tran_size = map_size
    else:
        tran_cells = pad_fraction*mesh_cells
        tran_size = pad_fraction*map_size

    cfield = np.zeros((tran_cells, tran_cells), dtype=complex)

    mesh_size = tran_size/tran_cells

    cfield = np.zeros((tran_cells, tran_cells), dtype=complex)

    # Look-up arrays for the wavenumbers
    # 2pi needed to convert frequency to angular frequency
    kx = np.fft.fftfreq(tran_cells, d=mesh_size)*2.*np.pi
    ky = kx  # The k in the y direction are identical to those in the x direction

    # Fill the map in Fourier space
    # TODO: Loops in python are bad. Is there a clever way of doing this?
    for i in range(tran_cells):
        for j in range(tran_cells):

            # Absolute wavenumber corresponding to cell in FFT array
            k = np.sqrt(kx[i]**2+ky[j]**2)

            if (i == 0 and j == 0):
                cfield[i, j] = mean_value
                # cfield[i, j] = 0.
            else:
                sigma = np.sqrt(power_spectrum(k, *args))/tran_size
                (x, y) = np.random.normal(0., sigma, size=2)
                cfield[i, j] = complex(x, y)

    # FFT
    cfield = np.fft.ifft2(cfield)
    cfield = cfield*tran_cells**2  # Normalisation

    # Convert to real
    field = np.zeros((mesh_cells, mesh_cells))
    # For non-periodic arrays we discard some
    field = np.real(cfield[0:mesh_cells, 0:mesh_cells])
    # field = field+mean

    return field

# Return a list of the integer coordinates of all local maxima in a 2D field


def find_field_peaks_2D(field: np.ndarray, nx: int, ny: int, periodic=True):

    # Initialise an empty list
    peaks = []

    # Loop over all cells in field
    for ix in range(nx):
        for iy in range(ny):

            # Get the central value of tjhe field
            central_value = field[ix, iy]

            # Get the coordinates of the neighbours
            neighbour_cells = neighbour_cells_2D(ix, iy, nx, ny, periodic)
            # print('Neighbour cells:', type(neighbour_cells), neighbour_cells)
            i, j = zip(*neighbour_cells)
            # print(ix, iy, i, j)

            if (all(neighbour_value < central_value for neighbour_value in field[i, j])):
                peaks.append([ix, iy])

    return peaks

# def downward_trajectory(i, j, field, nx, ny, periodic):
#
#   neighbour_cells = neighbour_cells_2D(i, j, nx, ny, periodic)
#
#   i, j = zip(*neighbour_cells)
#   field_values = field(neighbour_cells)

# Return a list of all cells that are neighbouring some integer cell coordinate


def neighbour_cells_2D(ix: int, iy: int, nx: int, ny: int, periodic=True):

    # Check if the cell is actually within the array
    if ((ix < 0) or (ix >= nx) or (iy < 0) or (iy >= ny)):
        raise ValueError('Cell coordinates are wrong')

    # Initialise an empty list
    cells = []

    # Name the cells sensibly
    ix_cell = ix
    ix_mini = ix-1
    ix_maxi = ix+1
    iy_cell = iy
    iy_mini = iy-1
    iy_maxi = iy+1

    # Deal with the edge cases for periodic and non-periodic fields sensibly
    if (ix_mini == -1):
        if (periodic):
            ix_mini = nx-1
        else:
            ix_mini = None
    if (iy_mini == -1):
        if (periodic):
            iy_mini = nx-1
        else:
            iy_mini = None
    if (ix_maxi == nx):
        if (periodic):
            ix_maxi = 0
        else:
            ix_maxi = None
    if (iy_maxi == ny):
        if (periodic):
            iy_maxi = 0
        else:
            iy_maxi = None

    # Add the neighbouring cells to the list
    for ixx in [ix_mini, ix_cell, ix_maxi]:
        for iyy in [iy_mini, iy_cell, iy_maxi]:
            if ((ixx != None) and (iyy != None)):
                cells.append([ixx, iyy])

    # Remove the initial cell from the list
    cells.remove([ix_cell, iy_cell])

    # Return a list of lists
    return cells


def compute_cross_spectrum(d1: np.ndarray, d2: np.ndarray, L: float,
                           kmin=None, kmax=None, nk=64, dimensionless=True,
                           eps_slop=1e-3) -> np.ndarray:
    '''
    Compute the cross spectrum between two real-space 2D density slices: d1 and d2
    If d1 = d2 then this will be the (auto) power spectrum, otherwise cross spectrum
    d1, d1: real-space density fields
    L: Box size [usually Mpc/h units in cosmology]
    kmin, kmax: Minimum/maximum wavenumbers
    nk: Number of k bins
    dimesionless: Compute Delta^2(k) rather than P(k)
    eps_slop: Ensure that all modes get counted (maybe unnecessary)
    TODO: Correct for binning by sharpening
    '''

    def _k_FFT(ix, iy, m, L):
        '''
        Get the wavenumber associated with the element in the FFT array
        ix, iy: indices in 2D array
        m: mesh size for FFT
        L: Box size [units]
        '''
        kx = ix if ix <= m/2 else m-ix
        ky = iy if iy <= m/2 else m-iy
        kx *= 2.*np.pi/L
        ky *= 2.*np.pi/L
        return kx, ky, np.sqrt(kx**2+ky**2)

    # Calculations
    m = d1.shape[0]  # Mesh size for array
    if kmin is None:
        kmin = 2.*np.pi/L  # Box frequency
    if kmax is None:
        kmax = m*np.pi/L  # Nyquist frequency

    # Bins for wavenumbers k
    kbin = np.logspace(np.log10(kmin), np.log10(
        kmax), nk+1)  # Note the +1 to have all bins
    kbin[0] *= (1.-eps_slop)
    kbin[-1] *= (1.+eps_slop)  # Avoid slop

    # Fourier transforms (renormalise for mesh size)
    # TODO: Use FFT that knows input fields are real
    dk1 = np.fft.fft2(d1)/m**2
    # Avoid a second FFT if possible
    dk2 = np.fft.fft2(d2)/m**2 if d1 is not d2 else dk1

    # Loop over Fourier arrays and accumulate power
    # TODO: These loops could be slow (rate limiting) in Python
    k = np.zeros(nk)
    power = np.zeros(nk)
    sigma = np.zeros(nk)
    nmodes = np.zeros(nk, dtype=int)
    for ix in range(m):
        for iy in range(m):
            _, _, kmod = _k_FFT(ix, iy, m, L)
            for i in range(nk):
                if kbin[i] <= kmod < kbin[i+1]:
                    k[i] += kmod
                    f = np.real(np.conj(dk1[ix, iy])*dk2[ix, iy])
                    power[i] += f
                    sigma[i] += f**2
                    nmodes[i] += 1
                    break

    # Averages over number of modes
    for i in range(nk):
        if nmodes[i] == 0:
            k[i] = np.sqrt(kbin[i+1]*kbin[i])
            power[i] = 0.
            sigma[i] = 0.
        else:
            k[i] /= nmodes[i]
            power[i] /= nmodes[i]
            sigma[i] /= nmodes[i]
            if nmodes[i] == 1:
                sigma[i] = 0.
            else:
                sigma[i] = np.sqrt(sigma[i]-power[i]**2)
                sigma[i] = sigma[i]*nmodes[i]/(nmodes[i]-1)
                sigma[i] = sigma[i]/np.sqrt(nmodes[i])

    # Create dimensionless spectra if desired
    if dimensionless:
        Dk = 2.*np.pi*(k*L/(2.*np.pi))**2
        power = power*Dk
        sigma = sigma*Dk

    return k, power, sigma, nmodes
