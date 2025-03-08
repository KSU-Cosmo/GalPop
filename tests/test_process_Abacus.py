import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
from astropy.table import Column

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the functions to test (adjust the import path as needed)
from process_Abacus import (
    process_Abacus_slab,
    process_Abacus_directory,
    save_results_fits,
    read_results_fits,
)

# ----- Mock Data and Fixtures -----

@pytest.fixture
def mock_compaso_catalog():
    """Create a mock CompaSOHaloCatalog for testing"""
    mock_cat = MagicMock()

    # Set up header
    mock_cat.header = {
        'BoxSizeHMpc': 500.0,
        'ParticleMassHMsun': 1.0e10,
        'H0': 70.0,
        'VelZSpace_to_kms': 100.0
    }
    # Set up halos (4 halos, 2 will pass the mass threshold)
    mock_cat.halos = {
        'N': Column(name='ParticleCounts', data=[100, 1000, 10000, 50]),  # Particle counts
        'x_L2com': Column(name='Positions', data=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]),  # Positions of halos
        'v_L2com': Column(name='Velocities', data=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]
        ]),  # Velocities of halos
        'npoutA': Column(name='SatelliteCounts', data=[3, 5, 10, 2]),  # Number of satellites per halo
        'sigmav3d_L2com': Column(name='VelocityDispersion', data=[100.0, 200.0, 300.0, 50.0])  # 3D velocity dispersion
    }
    # Set up subsamples
    mock_cat.subsamples = {
        'pos': Column(name='SubhaloPositions', data=np.random.rand(20, 3)),  # Random positions for subhalos
        'vel': Column(name='SubhaloVelocities', data=np.random.rand(20, 3) * 0.1)  # Random velocities for subhalos
    }
    return mock_cat

@pytest.fixture
def mock_results():
    """Create mock results dictionary for testing"""
    return {
        'halo': {
            'mass': np.array([1e13, 1e14]),
            'x': np.array([1.0, 2.0]),
            'y': np.array([3.0, 4.0]),
            'z': np.array([5.0, 6.0]),
            'sigma': np.array([0.1, 0.2]),
            'velocity': np.array([0.3, 0.4])
        },
        'subsample': {
            'mass': np.array([1e12, 1e12, 1e13]),
            'host_velocity': np.array([0.5, 0.6, 0.7]),
            'n_particles': np.array([100, 200, 300]),
            'x': np.array([7.0, 8.0, 9.0]),
            'y': np.array([10.0, 11.0, 12.0]),
            'z': np.array([13.0, 14.0, 15.0]),
            'velocity': np.array([0.8, 0.9, 1.0])
        }
    }

# ----- Tests for process_Abacus_slab -----

@patch('process_Abacus.CompaSOHaloCatalog')
def test_process_Abacus_slab(mock_CompaSO, mock_compaso_catalog):
    """Test processing of a single Abacus slab"""
    mock_CompaSO.return_value = mock_compaso_catalog
    result = process_Abacus_slab("dummy_slab.asdf", 12.0, 12.5, 5)
    _validate_result_structure(result)
    _validate_required_fields(result['halo'],
                            ['mass', 'x', 'y', 'z', 'sigma', 'velocity'],
                            'halo')
    _validate_required_fields(result['subsample'],
                            ['mass', 'host_velocity', 'n_particles', 'x', 'y', 'z', 'velocity'],
                            'subsample')
    _validate_array_shapes(result)

# ----- Tests for process_Abacus_directory -----

@patch('process_Abacus.glob.glob')
@patch('process_Abacus.process_Abacus_slab')
def test_process_Abacus_directory(mock_process_slab, mock_glob, mock_results):
    """Test processing of an Abacus directory"""
    mock_glob.return_value = ["slab1.asdf", "slab2.asdf"]
    mock_process_slab.return_value = mock_results
    
    result = process_Abacus_directory("/dummy/path/", 12.0, 12.5, 5)
    
    if mock_process_slab.call_count != 2:
        raise ValueError("process_Abacus_slab was not called twice")
    
    if 'halo' not in result:
        raise ValueError("Missing 'halo' key in result")
    if 'subsample' not in result:
        raise ValueError("Missing 'subsample' key in result")
    
    if len(result['halo']['mass']) != len(mock_results['halo']['mass']) * 2:
        raise ValueError("Unexpected length of halo mass array")
    if len(result['subsample']['mass']) != len(mock_results['subsample']['mass']) * 2:
        raise ValueError("Unexpected length of subsample mass array")

def _validate_fits_data(halo_data, subsample_data, mock_results):
    """Validate the data read from FITS file matches expected results"""
    if len(halo_data) != len(mock_results['halo']['mass']):
        raise ValueError("Halo data length mismatch")
    if len(subsample_data) != len(mock_results['subsample']['mass']):
        raise ValueError("Subsample data length mismatch")
    
    if not np.array_equal(halo_data['mass'], mock_results['halo']['mass']):
        raise ValueError("Halo mass arrays do not match")
    if not np.array_equal(subsample_data['mass'], mock_results['subsample']['mass']):
        raise ValueError("Subsample mass arrays do not match")

def test_save_and_read_fits(mock_results):
    """Test saving and reading results to/from FITS file"""
    fd, temp_filename = tempfile.mkstemp(suffix='.fits')
    os.close(fd)  # Close the file descriptor immediately
    
    try:
        save_results_fits(mock_results, temp_filename)

        if not os.path.exists(temp_filename):
            raise ValueError("FITS file was not created")
        
        halo_data, subsample_data = read_results_fits(temp_filename)
        _validate_fits_data(halo_data, subsample_data, mock_results)
    finally:
        os.unlink(temp_filename)  # Clean up the temporary file

def _validate_result_structure(result):
    """Validate the basic structure of the result dictionary"""
    if 'halo' not in result:
        raise ValueError("Missing 'halo' key in result")
    if 'subsample' not in result:
        raise ValueError("Missing 'subsample' key in result")

def _validate_required_fields(data, fields, data_type):
    """Validate that all required fields are present in the data"""
    for field in fields:
        if field not in data:
            raise ValueError(f"Missing {field} in {data_type} data")

def _validate_array_shapes(result):
    """Validate the shapes of arrays in the result"""
    for array in result['halo'].values():
        if array.shape != (2,):
            raise ValueError("Not all halo arrays have the expected shape")
    
    subsample_length = len(result['subsample']['mass'])
    for array in result['subsample'].values():
        if array.shape != (subsample_length,):
            raise ValueError("Subsample arrays have inconsistent shapes")
