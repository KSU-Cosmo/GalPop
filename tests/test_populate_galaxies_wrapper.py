import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import populate_galaxies_wrapper as pgw

return_xyz = {'x': [1, 2],
              'y': [3, 4],
              'z': [5, 6]}

@patch ('Main.populate_galaxies_julia', return_value=return_xyz)
def test_populate_galaxies():
    # Create mock halo and subsample dictionaries with test data
    h_mass = np.array([1e13, 2e13, 3e13]),  # Halo masses
    h_x = np.array([0, 10, 20]),  # x coordinates
    h_y = np.array([0, 10, 20]),  # y coordinates
    h_z = np.array([0, 10, 20]),  # z coordinates
    h_sigma = np.array([0.2, 0.3, 0.4]),  # sigma values
    h_velocity = np.array([10, 20, 30])  # Velocities
    
    s_mass = np.array([1e14, 2e14, 3e14]),  # Subsample masses
    s_host_velocity = np.array([1, 2, 3]),  # Host velocities
    s_n_particles = np.array([100, 200, 300]),  # Number of particles
    s_x = np.array([100, 110, 120]),  # x coordinates
    s_y = np.array([100, 110, 120]),  # y coordinates
    s_z = np.array([100, 110, 120]),  # z coordinates
    s_velocity = np.array([10, 20, 30])  # Velocities
    
    # Mock HOD parameters 
    # [lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s]
    hod_params = [
        np.log10(1e13),  # lnMcut
        0.2,             # sigma
        np.log10(1e14),  # lnM1
        1.0,             # kappa
        1.0,             # alpha
        0.1,             # alpha_c
        0.5              # alpha_s
    ]
    
    # Call the populate_galaxies function with separate h and s inputs
    galaxies = pgw.populate_galaxies(h_mass, h_x, h_y, h_z, h_velocity, h_sigma, s_mass, s_host_velocity, 
        s_n_particles, s_x, s_y, s_z, s_velocity, hod_params)

    # 1. Check that the output arrays have the same length
    assert galaxies['x'] == [1, 2]
    assert galaxies['y'] == [3, 4]
    assert galaxies['z'] == [5, 6]