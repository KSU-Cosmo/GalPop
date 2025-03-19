using Test
using HDF5
using GalPop  # Assuming save_to_hdf5 and load_from_hdf5 are accessible through GalPop

@testset "HDF5 IO Simple Test" begin
    # Create temporary file
    temp_filename = tempname() * ".h5"
    
    try
        # Create simple test data
        test_data = Dict(
            "halo" => Dict(
                "mass" => [1.0, 2.0, 3.0],
                "position" => [10.0 20.0 30.0; 15.0 25.0 35.0; 5.0 15.0 25.0]
            ),
            "galaxy" => Dict(
                "luminosity" => [5.5, 6.6, 7.7],
                "velocity" => [-1.0, 0.0, 1.0]
            )
        )
        
        # Save the data
        GalPop.save_to_hdf5(test_data, temp_filename)
        
        # Check file exists
        @test isfile(temp_filename)
        
        # Load the data
        loaded_data = GalPop.load_from_hdf5(temp_filename)
        
        # Check structure is preserved
        @test Set(keys(loaded_data)) == Set(keys(test_data))
        
        # Check each group has the right keys
        for group in keys(test_data)
            @test Set(keys(loaded_data[group])) == Set(keys(test_data[group]))
        end
        
        # Check data values match
        for group in keys(test_data)
            for key in keys(test_data[group])
                # Check array shape
                @test size(loaded_data[group][key]) == size(test_data[group][key])
                
                # Check values - using isapprox for floating point
                @test all(isapprox.(loaded_data[group][key], test_data[group][key]; rtol=1e-5))
            end
        end
        
    finally
        # Clean up
        isfile(temp_filename) && rm(temp_filename)
    end
end