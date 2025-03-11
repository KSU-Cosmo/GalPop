import numpy as np
import scipy as sp
import os

import julia
julia.Julia(compiled_modules=False)  # Initializes Julia

from julia import Main  # Imports Julia's Main module


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