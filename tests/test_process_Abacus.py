import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys
import tempfile
from astropy.table import Column

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from process_Abacus import (
    process_Abacus_slab,
    process_Abacus_directory,
    save_results_fits,
    read_results_fits,
)

@pytest.fixture
def mock_results():
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

@patch('process_Abacus.CompaSOHaloCatalog')
def test_process_Abacus_slab(mock_CompaSO):
    mock_CompaSO.return_value = MagicMock()
    result = process_Abacus_slab("dummy_slab.asdf", 12.0, 12.5, 5)

    if 'halo' not in result:
        raise ValueError("Missing 'halo' key in result")
    if 'subsample' not in result:
        raise ValueError("Missing 'subsample' key in result")
    
    for field in ['mass', 'x', 'y', 'z', 'sigma', 'velocity']:
        if field not in result['halo']:
            raise ValueError(f"Missing {field} in halo data")
    
    for field in ['mass', 'host_velocity', 'n_particles', 'x', 'y', 'z', 'velocity']:
        if field not in result['subsample']:
            raise ValueError(f"Missing {field} in subsample data")

@patch('process_Abacus.glob.glob')
@patch('process_Abacus.process_Abacus_slab')
def test_process_Abacus_directory(mock_process_slab, mock_glob, mock_results):
    mock_glob.return_value = ["slab1.asdf", "slab2.asdf"]
    mock_process_slab.return_value = mock_results
    result = process_Abacus_directory("/dummy/path/", 12.0, 12.5, 5)

    if mock_process_slab.call_count != 2:
        raise ValueError("process_Abacus_slab was not called twice")
    
    if 'halo' not in result or 'subsample' not in result:
        raise ValueError("Missing 'halo' or 'subsample' key in result")

    if len(result['halo']['mass']) != len(mock_results['halo']['mass']) * 2:
        raise ValueError("Unexpected length of halo mass array")
    
    if len(result['subsample']['mass']) != len(mock_results['subsample']['mass']) * 2:
        raise ValueError("Unexpected length of subsample mass array")

def test_save_and_read_fits(mock_results):
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        save_results_fits(mock_results, temp_filename)
        if not os.path.exists(temp_filename):
            raise ValueError("FITS file was not created")
        
        halo_data, subsample_data = read_results_fits(temp_filename)
        
        if len(halo_data) != len(mock_results['halo']['mass']):
            raise ValueError("Halo data length mismatch")
        
        if len(subsample_data) != len(mock_results['subsample']['mass']):
            raise ValueError("Subsample data length mismatch")
        
        np.testing.assert_array_equal(halo_data['mass'], mock_results['halo']['mass'])
        np.testing.assert_array_equal(subsample_data['mass'], mock_results['subsample']['mass'])
    
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
