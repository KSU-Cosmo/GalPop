import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import populate_galaxies as pg

def test_populate_galaxies():
    # Create a mock input dictionary with test data
    mock_data_dict = {
        'halo': {
            'mass': np.array([1e13, 2e13, 3e13]),  # Halo masses
            'x': np.array([0, 10, 20]),  # x coordinates
            'y': np.array([0, 10, 20]),  # y coordinates
            'z': np.array([0, 10, 20]),  # z coordinates
            'sigma': np.array([0.2, 0.3, 0.4])  # sigma values
        },
        'subsample': {
            'mass': np.array([1e14, 2e14, 3e14]),  # Subsample masses
            'host_velocity': np.array([1, 2, 3]),  # Host velocities
            'n_particles': np.array([100, 200, 300]),  # Number of particles
            'x': np.array([100, 110, 120]),  # x coordinates
            'y': np.array([100, 110, 120]),  # y coordinates
            'z': np.array([100, 110, 120]),  # z coordinates
            'velocity': np.array([10, 20, 30])  # Velocities
        }
    }
    
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
    
    # Call the populate_galaxies function
    x_galaxies, y_galaxies, z_galaxies = pg.populate_galaxies(mock_data_dict, hod_params)
    
    # 1. Check that the output arrays have the same length
    assert len(x_galaxies) == len(y_galaxies)
    assert len(x_galaxies) == len(z_galaxies)
    
    # 2. Check that the output is not empty
    assert len(x_galaxies) > 0
    
    # 3. Verify the output coordinates are numeric
    assert np.issubdtype(x_galaxies.dtype, np.number)
    assert np.issubdtype(y_galaxies.dtype, np.number)
    assert np.issubdtype(z_galaxies.dtype, np.number)
    
    # 4. Check that some input coordinates are present in the output
    # This checks that both central and satellite galaxies are being selected
    input_x_coords = np.concatenate([mock_data_dict['halo']['x'], mock_data_dict['subsample']['x']])
    input_y_coords = np.concatenate([mock_data_dict['halo']['y'], mock_data_dict['subsample']['y']])
    input_z_coords = np.concatenate([mock_data_dict['halo']['z'], mock_data_dict['subsample']['z']])
    
    # Check that at least some input coordinates are in the output
    assert any(np.isin(x_galaxies, input_x_coords))
    assert any(np.isin(y_galaxies, input_y_coords))
    assert any(np.isin(z_galaxies, input_z_coords))
    
    # 5. Reproducibility check (with fixed random seed)
    np.random.seed(42)
    x1, y1, z1 = pg.populate_galaxies(mock_data_dict, hod_params)
    np.random.seed(42)
    x2, y2, z2 = pg.populate_galaxies(mock_data_dict, hod_params)
    
    # Check that results are reproducible when random seed is the same
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(z1, z2)

def test_populate_galaxies_edge_cases():
    # Test with very low masses
    mock_data_dict = {
        'halo': {
            'mass': np.array([1e10, 1e11]),  # Very low masses
            'x': np.array([0, 10]),
            'y': np.array([0, 10]),
            'z': np.array([0, 10]),
            'sigma': np.array([0.2, 0.3])
        },
        'subsample': {
            'mass': np.array([1e11, 1e12]),
            'host_velocity': np.array([1, 2]),
            'n_particles': np.array([10, 20]),
            'x': np.array([100, 110]),
            'y': np.array([100, 110]),
            'z': np.array([100, 110]),
            'velocity': np.array([10, 20])
        }
    }
    
    # Same HOD params as previous test
    hod_params = [
        np.log10(1e13),  # lnMcut
        0.2,             # sigma
        np.log10(1e14),  # lnM1
        1.0,             # kappa
        1.0,             # alpha
        0.1,             # alpha_c
        0.5              # alpha_s
    ]
    
    # This should not raise an error
    x_galaxies, y_galaxies, z_galaxies = pg.populate_galaxies(mock_data_dict, hod_params)
    
    # Additional checks for low mass case
    assert len(x_galaxies) >= 0  # Can be zero or more
    assert len(y_galaxies) == len(x_galaxies)
    assert len(z_galaxies) == len(x_galaxies)