import numpy as np
import scipy as sp
import os

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

current_dir = os.path.dirname(os.path.abspath(__file__))
julia_file = os.path.join(current_dir, "populate_galaxies.jl")
Main.include(julia_file)

# Python wrapper for the Julia function
def populate_galaxies(h, s, HODparams, rsd=True, Lmin=-1000, Lmax=1000):
    """
    Python wrapper for the Julia implementation of populate_galaxies.
    
    Parameters
    ----------
    h : dict
        Dictionary containing halo properties
    s : dict
        Dictionary containing subhalo properties
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
    
    # Unpack halo data and ensure float64 type
    h_mass = np.asarray(h['mass'], dtype=np.float64)
    h_x = np.asarray(h['x'], dtype=np.float64)
    h_y = np.asarray(h['y'], dtype=np.float64)
    h_z = np.asarray(h['z'], dtype=np.float64)
    h_velocity = np.asarray(h['velocity'], dtype=np.float64)
    h_sigma = np.asarray(h['sigma'], dtype=np.float64)
    
    # Unpack subsample data and ensure float64 type
    s_mass = np.asarray(s['mass'], dtype=np.float64)
    s_host_velocity = np.asarray(s['host_velocity'], dtype=np.float64)
    s_n_particles = np.asarray(s['n_particles'], dtype=np.float64)
    s_x = np.asarray(s['x'], dtype=np.float64)
    s_y = np.asarray(s['y'], dtype=np.float64)
    s_z = np.asarray(s['z'], dtype=np.float64)
    s_velocity = np.asarray(s['velocity'], dtype=np.float64)
    
    # Unpack HOD parameters
    lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s = HODparams
    
    # Call the Julia function
    x_gal, y_gal, z_gal = Main.populate_galaxies_julia(
        h_mass, h_x, h_y, h_z, h_velocity, h_sigma,
        s_mass, s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
        lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s,
        rsd, float(Lmin), float(Lmax)
    )
    
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