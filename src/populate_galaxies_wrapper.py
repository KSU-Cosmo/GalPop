import numpy as np
import scipy as sp
import os

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

current_dir = os.path.dirname(os.path.abspath(__file__))
julia_file = os.path.join(current_dir, "populate_galaxies.jl")
Main.include(julia_file)

# Python wrapper for the Julia function that accepts individual arrays
def populate_galaxies(
    h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
    s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
    HODparams, rsd=True, Lmin=-1000, Lmax=1000
):
    """
    Python wrapper for the Julia implementation of populate_galaxies that accepts individual arrays.
    
    Parameters
    ----------
    h_mass : array-like
        Halo masses
    h_x, h_y, h_z : array-like
        Halo coordinates
    h_velocity : array-like
        Halo velocities
    h_sigma : array-like
        Halo velocity dispersions
    s_mass : array-like
        Subsample masses
    s_host_velocity : array-like
        Subsample host velocities
    s_n_particles : array-like
        Number of particles in each subsample
    s_x, s_y, s_z : array-like
        Subsample coordinates
    s_velocity : array-like
        Subsample velocities
    HODparams : sequence
        Sequence of HOD parameters (lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s)
    rsd : bool, optional
        Whether to apply redshift-space distortions, by default True
    Lmin : float, optional
        Minimum box coordinate, by default -1000
    Lmax : float, optional
        Maximum box coordinate, by default 1000
        
    Returns
    -------
    dict
        Dictionary of galaxy positions with keys 'x', 'y', 'z'
    """
    
    print("passing arrays")
    # Before your conversion code, add:
    for name, arr in [
    ('h_mass', h_mass), ('h_x', h_x), ('h_y', h_y), ('h_z', h_z), 
    ('h_velocity', h_velocity), ('h_sigma', h_sigma),
    ('s_mass', s_mass), ('s_host_velocity', s_host_velocity), 
    ('s_n_particles', s_n_particles), ('s_x', s_x), 
    ('s_y', s_y), ('s_z', s_z), ('s_velocity', s_velocity)
    ]:
        print(f"{name}: type={type(arr)}, dtype={getattr(arr, 'dtype', None)}, shape={getattr(arr, 'shape', None)}")
    # Ensure all arrays are float64 type
    '''
    h_mass = np.asarray(h_mass, dtype=np.float64)
    h_x = np.asarray(h_x, dtype=np.float64)
    h_y = np.asarray(h_y, dtype=np.float64)
    h_z = np.asarray(h_z, dtype=np.float64)
    h_velocity = np.asarray(h_velocity, dtype=np.float64)
    h_sigma = np.asarray(h_sigma, dtype=np.float64)
    
    s_mass = np.asarray(s_mass, dtype=np.float64)
    s_host_velocity = np.asarray(s_host_velocity, dtype=np.float64)
    s_n_particles = np.asarray(s_n_particles, dtype=np.float64)
    s_x = np.asarray(s_x, dtype=np.float64)
    s_y = np.asarray(s_y, dtype=np.float64)
    s_z = np.asarray(s_z, dtype=np.float64)
    s_velocity = np.asarray(s_velocity, dtype=np.float64)
    '''
    print("unpacking HOD parameters")
    # Unpack HOD parameters
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s = HODparams
    
    print("call the Julia function")
    # Call the Julia function
    x_gal, y_gal, z_gal = Main.populate_galaxies_julia(
        h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
        s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
        lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s,
        rsd, float(Lmin), float(Lmax)
    )
    
    print("convert the arrays back to python")
    # Convert Julia arrays to NumPy arrays
    x_gal = np.array(x_gal)
    y_gal = np.array(y_gal)
    z_gal = np.array(z_gal)
    
    # Return dictionary of galaxy positions
    return {
        'x': x_gal,
        'y': y_gal,
        'z': z_gal
    }