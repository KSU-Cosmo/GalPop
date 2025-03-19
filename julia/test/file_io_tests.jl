using Test
using HDF5
using GalPop  # Assuming save_to_hdf5 and load_from_hdf5 are accessible through GalPop

@testset "HDF5 IO Simple Test" begin
    # Create temporary file
    temp_filename = tempname() * ".h5"

    try
        # Create simple test data with fixed sizes
        test_data = Dict(
            "halo" => Dict("mass" => [1.0, 2.0, 3.0], "position" => [10.0, 20.0, 30.0]),
            "galaxy" =>
                Dict("luminosity" => [5.5, 6.6, 7.7], "velocity" => [-1.0, 0.0, 1.0]),
        )

        # Write directly to HDF5 file without using save_to_hdf5
        h5open(temp_filename, "w") do file
            for (group_name, group_data) in test_data
                # Create group
                g = create_group(file, group_name)

                # Write datasets in the group
                for (dataset_name, dataset_values) in group_data
                    g[dataset_name] = dataset_values
                end
            end
        end

        # Check file exists
        @test isfile(temp_filename)

        # Load the data using load_from_hdf5
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
                @test all(
                    isapprox.(loaded_data[group][key], test_data[group][key]; rtol=1e-5)
                )
            end
        end

    finally
        # Clean up
        isfile(temp_filename) && rm(temp_filename)
    end
end
