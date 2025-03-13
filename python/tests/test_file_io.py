import pytest
import numpy as np
import os
import tempfile

# Import the functions to test
# Assuming your functions are in a module called hdf5_io
# Adjust the import as needed for your project structure
from galpop.file_io import save_to_hdf5, load_from_hdf5


@pytest.fixture
def sample_data():
    """Generate a sample data structure similar to the original."""
    # Create random data with similar structure
    return {
        "halo": {
            "mass": np.random.rand(100) * 1e12,
            "x": np.array(np.random.rand(100) * 1000, dtype=np.float32),
            "y": np.array(np.random.rand(100) * 2000 - 1000, dtype=np.float32),
            "z": np.array(np.random.rand(100) * 2000 - 1000, dtype=np.float32),
            "sigma": np.array(np.random.rand(100) * 1e-7, dtype=np.float32),
            "velocity": np.array(np.random.rand(100) * 1e-6 - 5e-7, dtype=np.float32),
        },
        "subsample": {
            "mass": np.random.rand(50) * 1e13,
            "host_velocity": np.array(np.random.rand(50) * 1e-6, dtype=np.float32),
            "n_particles": np.array(np.random.randint(10, 40, 50), dtype=np.uint32),
            "x": np.array(np.random.rand(50) * 1000, dtype=np.float32),
            "y": np.array(np.random.rand(50) * 2000 - 1000, dtype=np.float32),
            "z": np.array(np.random.rand(50) * 1000 - 500, dtype=np.float32),
            "velocity": np.array(np.random.rand(50) * 2e-6 - 1e-6, dtype=np.float32),
        },
    }


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_filename = f.name

    # Return filename and ensure it gets cleaned up after the test
    yield temp_filename
    if os.path.exists(temp_filename):
        os.remove(temp_filename)


def test_save_to_hdf5(sample_data, temp_hdf5_file):
    """Test that save_to_hdf5 creates a file with the expected structure."""
    # Save the data
    save_to_hdf5(sample_data, temp_hdf5_file)

    # Check that file exists
    assert os.path.exists(temp_hdf5_file), "HDF5 file was not created"

    # Check file size is reasonable (non-zero)
    assert os.path.getsize(temp_hdf5_file) > 0, "HDF5 file is empty"


def test_load_from_hdf5(sample_data, temp_hdf5_file):
    """Test that load_from_hdf5 correctly loads the saved data."""
    # Save the data first
    save_to_hdf5(sample_data, temp_hdf5_file)

    # Load the data
    loaded_data = load_from_hdf5(temp_hdf5_file)

    # Check structure is preserved
    assert set(loaded_data.keys()) == set(sample_data.keys()), "Top-level keys don't match"

    # Check each group has the right keys
    for group in sample_data.keys():
        assert set(loaded_data[group].keys()) == set(
            sample_data[group].keys()
        ), f"Keys in {group} group don't match"


def test_data_integrity(sample_data, temp_hdf5_file):
    """Test that the data values are preserved correctly."""
    # Save and load the data
    save_to_hdf5(sample_data, temp_hdf5_file)
    loaded_data = load_from_hdf5(temp_hdf5_file)

    # Check each array for equality
    for group in sample_data.keys():
        for key in sample_data[group].keys():
            # Check array shape
            assert (
                loaded_data[group][key].shape == sample_data[group][key].shape
            ), f"Shape mismatch for {group}/{key}"

            # Check data type
            assert (
                loaded_data[group][key].dtype == sample_data[group][key].dtype
            ), f"Data type mismatch for {group}/{key}"

            # Check values - using allclose for floating point
            if np.issubdtype(sample_data[group][key].dtype, np.floating):
                assert np.allclose(
                    loaded_data[group][key], sample_data[group][key]
                ), f"Values don't match for {group}/{key}"
            else:
                assert np.array_equal(
                    loaded_data[group][key], sample_data[group][key]
                ), f"Values don't match for {group}/{key}"


def test_large_data_handling():
    """Test handling of larger datasets."""
    # Create a larger dataset (adjust size as needed for your tests)
    large_data = {
        "halo": {
            "mass": np.random.rand(10000) * 1e12,
            "positions": np.random.rand(10000, 3),  # 3D positions
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_filename = f.name

    try:
        # Test saving and loading
        save_to_hdf5(large_data, temp_filename)
        loaded_data = load_from_hdf5(temp_filename)

        # Verify key parts of the data
        assert loaded_data["halo"]["mass"].shape == large_data["halo"]["mass"].shape
        assert np.allclose(loaded_data["halo"]["positions"], large_data["halo"]["positions"])
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_julia_compatibility():
    """
    Test that saved files are compatible with Julia.
    This test is a placeholder - actual Julia compatibility
    would need to be tested in a Julia environment.
    """
    # Create test data
    test_data = {"halo": {"mass": np.array([1.0, 2.0, 3.0]), "x": np.array([10.0, 20.0, 30.0])}}

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_filename = f.name

    try:
        # Save the data in a standard HDF5 format
        save_to_hdf5(test_data, temp_filename)

        # The following is a comment describing how to test in Julia
        # This would be run in a Julia environment:
        """
        using HDF5
        using Test
        
        @testset "Julia compatibility test" begin
            data = load_from_hdf5("$temp_filename")
            @test haskey(data, "halo")
            @test haskey(data["halo"], "mass")
            @test data["halo"]["mass"] ≈ [1.0, 2.0, 3.0]
            @test data["halo"]["x"] ≈ [10.0, 20.0, 30.0]
        end
        """

        # For this Python test, we'll just verify the file exists and has content
        assert os.path.exists(temp_filename)
        assert os.path.getsize(temp_filename) > 0
    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
