import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
from astropy.io import fits
from astropy.table import Table
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
    # Total of 20 subhalos (sum of npoutA)
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
    # Set up the mock
    mock_CompaSO.return_value = mock_compaso_catalog
    
    # Call the function
    result = process_Abacus_slab("dummy_slab.asdf", 12.0, 12.5, 5)
    
    # Check the result structure
    assert 'halo' in result
    assert 'subsample' in result
    
    # Check that halo fields exist
    for field in ['mass', 'x', 'y', 'z', 'sigma', 'velocity']:
        assert field in result['halo']
    
    # Check that subsample fields exist
    for field in ['mass', 'host_velocity', 'n_particles', 'x', 'y', 'z', 'velocity']:
        assert field in result['subsample']
    
    # Check that the masks were applied correctly
    # Given our mock data and thresholds, we expect 2 halos to pass the mass threshold
    assert all(array.shape == (2,) for array in result['halo'].values())
    
    # The exact number of subsamples will depend on the masking logic
    subsample_length = len(result['subsample']['mass'])
    assert all(array.shape == (subsample_length,) for array in result['subsample'].values())

# ----- Tests for process_Abacus_directory -----

@patch('process_Abacus.glob.glob')
@patch('process_Abacus.process_Abacus_slab')
def test_process_Abacus_directory(mock_process_slab, mock_glob, mock_results):
    """Test processing of an Abacus directory"""
    # Set up mocks
    mock_glob.return_value = ["slab1.asdf", "slab2.asdf"]
    mock_process_slab.return_value = mock_results
    
    # Call the function
    result = process_Abacus_directory("/dummy/path/", 12.0, 12.5, 5)
    
    # Check that process_Abacus_slab was called the expected number of times
    assert mock_process_slab.call_count == 2
    
    # Check that the result has the expected structure
    assert 'halo' in result
    assert 'subsample' in result
    
    # Since we return the same mock_results twice, the lengths should double
    assert len(result['halo']['mass']) == len(mock_results['halo']['mass']) * 2
    assert len(result['subsample']['mass']) == len(mock_results['subsample']['mass']) * 2

def test_process_Abacus_directory_error_handling():
    """Test error handling in process_Abacus_directory"""
    with patch('process_Abacus.glob.glob') as mock_glob:
        mock_glob.return_value = ["slab1.asdf", "slab2.asdf"]
        
        with patch('process_Abacus.process_Abacus_slab') as mock_process_slab:
            # First call succeeds, second call raises an exception
            mock_process_slab.side_effect = [
                {'halo': {'mass': np.array([1e13]), 'x': np.array([1.0]), 'y': np.array([2.0]), 
                          'z': np.array([3.0]), 'sigma': np.array([0.1]), 'velocity': np.array([0.2])},
                 'subsample': {'mass': np.array([1e12]), 'host_velocity': np.array([0.3]), 
                              'n_particles': np.array([100]), 'x': np.array([4.0]), 
                              'y': np.array([5.0]), 'z': np.array([6.0]), 'velocity': np.array([0.4])}},
                Exception("Test error")
            ]
            
            # Should not raise an exception
            result = process_Abacus_directory("/dummy/path/", 12.0, 12.5, 5)
            
            # Should still have the results from the first slab
            assert len(result['halo']['mass']) == 1
            assert len(result['subsample']['mass']) == 1

# ----- Tests for save_results_fits and read_results_fits -----

def test_save_and_read_fits(mock_results):
    """Test saving and reading results to/from FITS file"""
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # Save the results
        save_results_fits(mock_results, temp_filename)
        
        # Check the file was created
        assert os.path.exists(temp_filename)
        
        # Read the results back
        halo_data, subsample_data = read_results_fits(temp_filename)
        
        # Check the data was read correctly
        assert len(halo_data) == len(mock_results['halo']['mass'])
        assert len(subsample_data) == len(mock_results['subsample']['mass'])
        
        # Check the field values match
        np.testing.assert_array_equal(halo_data['mass'], mock_results['halo']['mass'])
        np.testing.assert_array_equal(subsample_data['mass'], mock_results['subsample']['mass'])
        
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

# ----- Integration Tests -----

def test_process_and_save_workflow():
    """Integration test for the full workflow"""
    with patch('process_Abacus.CompaSOHaloCatalog') as mock_CompaSO, \
         patch('process_Abacus.glob.glob') as mock_glob:
        
        # Set up the mock catalog
        mock_catalog = MagicMock()
        mock_catalog.header = {
            'BoxSizeHMpc': 500.0,
            'ParticleMassHMsun': 1.0e10,
            'H0': 70.0,
            'VelZSpace_to_kms': 100.0
        }
        
        # Create simple test data with astropy Columns
        mock_catalog.halos = {
            'N': Column(data=np.array([1000, 2000]), name='N'),
            'x_L2com': Column(data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), name='x_L2com'),
            'v_L2com': Column(data=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), name='v_L2com'),
            'npoutA': Column(data=np.array([2, 3]), name='npoutA'),
            'sigmav3d_L2com': Column(data=np.array([200.0, 300.0]), name='sigmav3d_L2com')
        }
        
        mock_catalog.subsamples = {
            'pos': Column(data=np.random.rand(5, 3), name='pos'),
            'vel': Column(data=np.random.rand(5, 3) * 0.1, name='vel')
        }
        
        # Ensure the mock is returned for each slab
        mock_CompaSO.return_value = mock_catalog
        mock_glob.return_value = ["slab1.asdf"]
        
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Process the directory
            print("Processing directory...")
            results = process_Abacus_directory("/dummy/path/", 12.0, 12.5, 5)
            print(f"Results structure: {results.keys()}")
            print(f"Halo data length: {len(results['halo']['mass'])}")
            print(f"Subsample data length: {len(results['subsample']['mass'])}")
            
            # Save the results
            print("Saving results...")
            save_results_fits(results, temp_filename)
            
            # Read the results back
            print("Reading results...")
            halo_data, subsample_data = read_results_fits(temp_filename)
            print(f"Read halo data length: {len(halo_data)}")
            print(f"Read subsample data length: {len(subsample_data)}")
            
            # Check we got something back
            assert len(halo_data) > 0
            assert len(subsample_data) > 0
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
