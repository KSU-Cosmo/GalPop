import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Import our helper first to ensure Julia is set up correctly
from tests.julia_helper import Main, create_mock_return

# Fix the import path
srcpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(srcpath)

# Now import the module
import populate_galaxies_wrapper as pgw

def test_populate_galaxies():
    # Get mock return value from helper
    ret_xyz = create_mock_return()
    
    # Create mock halo and subsample dictionaries with test data
    h_mass = np.array([1e13, 2e13, 3e13], dtype=np.float32)  # Halo masses
    h_x = np.array([0, 10, 20], dtype=np.float32)  # x coordinates
    h_y = np.array([0, 10, 20], dtype=np.float32)  # y coordinates
    h_z = np.array([0, 10, 20], dtype=np.float32)  # z coordinates
    h_sigma = np.array([0.2, 0.3, 0.4], dtype=np.float32)  # sigma values
    h_velocity = np.array([10, 20, 30], dtype=np.float32)  # Velocities
    
    # Subsample masses
    s_mass = np.array([1e14, 2e14, 3e14], dtype=np.float32)
    # Host velocities
    s_host_velocity = np.array([1, 2, 3], dtype=np.float32)
    # Number of particles
    s_n_particles = np.array([100, 200, 300], dtype=np.int32)
    s_x = np.array([100, 110, 120], dtype=np.float32)  # x coordinates
    s_y = np.array([100, 110, 120], dtype=np.float32)  # y coordinates
    s_z = np.array([100, 110, 120], dtype=np.float32)  # z coordinates
    s_velocity = np.array([10, 20, 30], dtype=np.float32)  # Velocities
    
    # Mock HOD parameters
    # [lnMcut, sigma, lnM1, kappa, alpha, alpha_c, alpha_s]
    hod_params = [
        np.log10(1e13),  # lnMcut
        0.2,  # sigma
        np.log10(1e14),  # lnM1
        1.0,  # kappa
        1.0,  # alpha
        0.1,  # alpha_c
        0.5  # alpha_s
    ]
    
    # Alternative approach: monkey patch the function at module level instead of using patch.object
    original_func = pgw.Main.populate_galaxies_julia
    
    try:
        # Replace the function with a mock that returns our fixed values
        pgw.Main.populate_galaxies_julia = MagicMock(return_value=(ret_xyz['x'], ret_xyz['y'], ret_xyz['z']))
        
        # Call the populate_galaxies function with separate h and s inputs
        galaxies = pgw.populate_galaxies(
            h_mass, h_x, h_y, h_z, h_velocity, h_sigma, s_mass,
            s_host_velocity, s_n_particles, s_x, s_y, s_z, s_velocity,
            hod_params
        )
        
        # Add assertions to ensure the test actually validates something
        assert np.array_equal(galaxies['x'], ret_xyz['x'])
        assert np.array_equal(galaxies['y'], ret_xyz['y'])
        assert np.array_equal(galaxies['z'], ret_xyz['z'])
    finally:
        # Restore the original function
        pgw.Main.populate_galaxies_julia = original_func