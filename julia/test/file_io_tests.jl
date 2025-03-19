using Test
using HDF5
using Random
using GalPop
using file_io

# Assuming your functions are in a module called HDF5IO
# include("path/to/hdf5_io.jl")  # Uncomment and adjust as needed
# using .HDF5IO: save_to_hdf5, load_from_hdf5

# If not using a module, define the functions directly:
# Paste your save_to_hdf5 and load_from_hdf5 functions here

"""
Generate a sample data structure similar to the original.
"""
function create_sample_data()
    # Set random seed for reproducibility
    Random.seed!(42)
    
    return Dict(
        "halo" => Dict(
            "mass" => rand(100) .* 1e12,
            "x" => Float32.(rand(100) .* 1000),
            "y" => Float32.(rand(100) .* 2000 .- 1000),
            "z" => Float32.(rand(100) .* 2000 .- 1000),
            "sigma" => Float32.(rand(100) .* 1e-7),
            "velocity" => Float32.(rand(100) .* 1e-6 .- 5e-7)
        ),
        "subsample" => Dict(
            "mass" => rand(50) .* 1e13,
            "host_velocity" => Float32.(rand(50) .* 1e-6),
            "n_particles" => UInt32.(rand(10:40, 50)),
            "x" => Float32.(rand(50) .* 1000),
            "y" => Float32.(rand(50) .* 2000 .- 1000),
            "z" => Float32.(rand(50) .* 1000 .- 500),
            "velocity" => Float32.(rand(50) .* 2e-6 .- 1e-6)
        )
    )
end

@testset "HDF5 IO Tests" begin
    # Create temporary file
    temp_filename = tempname() * ".h5"
    
    try
        # Create sample data
        sample_data = create_sample_data()
        
        @testset "Save to HDF5" begin
            # Test saving
            save_to_hdf5(sample_data, temp_filename)
            
            # Check file exists
            @test isfile(temp_filename)
            
            # Check file size is reasonable
            @test filesize(temp_filename) > 0
        end
        
        @testset "Load from HDF5" begin
            # Load the data
            loaded_data = load_from_hdf5(temp_filename)
            
            # Check structure is preserved
            @test Set(keys(loaded_data)) == Set(keys(sample_data))
            
            # Check each group has the right keys
            for group in keys(sample_data)
                @test Set(keys(loaded_data[group])) == Set(keys(sample_data[group]))
            end
        end
        
        @testset "Data Integrity" begin
            # Load the data
            loaded_data = load_from_hdf5(temp_filename)
            
            # Check each array for equality
            for group in keys(sample_data)
                for key in keys(sample_data[group])
                    # Check array shape
                    @test size(loaded_data[group][key]) == size(sample_data[group][key])
                    
                    # Check values - using isapprox for floating point
                    if eltype(sample_data[group][key]) <: AbstractFloat
                        @test all(isapprox.(loaded_data[group][key], sample_data[group][key]; rtol=1e-5))
                    else
                        @test loaded_data[group][key] == sample_data[group][key]
                    end
                end
            end
        end
        
        @testset "Large Data Handling" begin
            # Create a larger dataset
            large_data = Dict(
                "halo" => Dict(
                    "mass" => rand(10000) .* 1e12,
                    "positions" => rand(10000, 3)  # 3D positions
                )
            )
            
            # Test saving and loading
            large_filename = tempname() * ".h5"
            try
                save_to_hdf5(large_data, large_filename)
                loaded_large_data = load_from_hdf5(large_filename)
                
                # Verify key parts of the data
                @test size(loaded_large_data["halo"]["mass"]) == size(large_data["halo"]["mass"])
                @test all(isapprox.(loaded_large_data["halo"]["positions"], large_data["halo"]["positions"]; rtol=1e-5))
            finally
                # Clean up
                isfile(large_filename) && rm(large_filename)
            end
        end
        
        @testset "Python Compatibility" begin
            # Create simple test data
            test_data = Dict(
                "halo" => Dict(
                    "mass" => [1.0, 2.0, 3.0],
                    "x" => [10.0, 20.0, 30.0]
                )
            )
            
            # Save the data
            compatibility_filename = tempname() * ".h5"
            try
                save_to_hdf5(test_data, compatibility_filename)
                
                # This is just a comment about how to verify in Python:
                """
                import h5py
                import numpy as np
                
                with h5py.File("$compatibility_filename", "r") as f:
                    assert "halo" in f
                    assert "mass" in f["halo"]
                    assert np.allclose(f["halo"]["mass"][:], np.array([1.0, 2.0, 3.0]))
                    assert np.allclose(f["halo"]["x"][:], np.array([10.0, 20.0, 30.0]))
                """
                
                # Just verify the file exists
                @test isfile(compatibility_filename)
                @test filesize(compatibility_filename) > 0
            finally
                # Clean up
                isfile(compatibility_filename) && rm(compatibility_filename)
            end
        end
        
    finally
        # Clean up
        isfile(temp_filename) && rm(temp_filename)
    end
end