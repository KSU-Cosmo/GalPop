# File: julia/test/hod_tests.jl

using Test
using Random
using Statistics
using GalPop

@testset "HOD Module Tests" begin
    # Set random seed for reproducibility
    Random.seed!(123)
    
    # Create test data
    n_halos = 1000
    n_subhalos = 5000
    
    # Generate halo data
    h_mass = Float32.(10.0 .^ rand(11.5:0.01:14.5, n_halos))
    h_x = rand(Float32, n_halos) .* 1000.0
    h_y = rand(Float32, n_halos) .* 1000.0
    h_z = rand(Float32, n_halos) .* 1000.0
    h_velocity = randn(Float32, n_halos) .* 300.0
    h_sigma = rand(Float32, n_halos) .* 200.0 .+ 100.0
    
    halos = (
        mass = h_mass,
        x = h_x,
        y = h_y,
        z = h_z,
        velocity = h_velocity,
        sigma = h_sigma
    )
    
    # Generate subhalo data
    s_mass = Float32.(10.0 .^ rand(10.5:0.01:13.5, n_subhalos))
    s_host_velocity = randn(Float32, n_subhalos) .* 300.0
    s_n_particles = Int32.(rand(10:1000, n_subhalos))
    s_x = rand(Float32, n_subhalos) .* 1000.0
    s_y = rand(Float32, n_subhalos) .* 1000.0
    s_z = rand(Float32, n_subhalos) .* 1000.0
    s_velocity = randn(Float32, n_subhalos) .* 400.0
    
    subhalos = (
        mass = s_mass,
        host_velocity = s_host_velocity,
        n_particles = s_n_particles,
        x = s_x,
        y = s_y,
        z = s_z,
        velocity = s_velocity
    )
    
    # Test HOD parameter sets
    test_params = [
        # Standard parameters
        (
            lnMcut = 12.0, 
            sigma = 0.5, 
            lnM1 = 13.5, 
            kappa = 1.0, 
            alpha = 1.0, 
            alpha_c = 0.3, 
            alpha_s = 1.0,
            rsd = true,
            Lmin = 0.0,
            Lmax = 1000.0
        ),
        # Alternative parameters
        (
            lnMcut = 12.5, 
            sigma = 0.3, 
            lnM1 = 14.0, 
            kappa = 0.9, 
            alpha = 1.1, 
            alpha_c = 0.5, 
            alpha_s = 0.8,
            rsd = false,
            Lmin = 0.0,
            Lmax = 500.0
        )
    ]
    
    @testset "calculate_p_cen function" begin
        # Test with known values
        test_masses = Float32[1e11, 1e12, 1e13, 1e14]
        Mcut = 1e12
        sigma = 0.5
        
        p_values = GalPop.calculate_p_cen(test_masses, Mcut, sigma)
        
        # Lower mass should have lower probability
        @test p_values[1] < 0.5
        # Equal mass should have probability 0.5
        @test isapprox(p_values[2], 0.5, atol=1e-6)
        # Higher mass should have higher probability
        @test p_values[3] > 0.5
        @test p_values[4] > p_values[3]
        
        # Test with different sigma
        p_values_wider = GalPop.calculate_p_cen(test_masses, Mcut, 1.0)
        
        # Wider sigma should give higher probability for lower mass
        @test p_values_wider[1] > p_values[1]
    end
    
    @testset "calculate_n_sat function" begin
        # Test with masses below threshold
        n_sat = GalPop.calculate_n_sat(Float32(9e11), 1e12, 1e13, 1.0, 1.0, 0.5)
        @test n_sat == 0.0
        
        # Test with masses above threshold
        n_sat = GalPop.calculate_n_sat(Float32(2e12), 1e12, 1e13, 1.0, 1.0, 0.5)
        @test n_sat > 0.0
        @test isapprox(n_sat, 0.5 * (1e12 / 1e13), atol=1e-6)
        
        # Test with different alpha
        n_sat_alpha2 = GalPop.calculate_n_sat(Float32(2e12), 1e12, 1e13, 2.0, 1.0, 0.5)
        @test n_sat_alpha2 < n_sat  # Higher alpha should give lower n_sat for mass < M1
    end
    
    @testset "populate_galaxies function" begin
        for params in test_params
            # Run the galaxy population
            galaxies = GalPop.populate_galaxies(halos, subhalos, params)
            
            # Check the return structure
            @test haskey(galaxies, :x)
            @test haskey(galaxies, :y)
            @test haskey(galaxies, :z)
            @test haskey(galaxies, :count)
            
            # Check dimensions
            @test length(galaxies.x) == galaxies.count
            @test length(galaxies.y) == galaxies.count
            @test length(galaxies.z) == galaxies.count
            
            # Check that we have a reasonable number of galaxies
            @test 0 < galaxies.count < (n_halos + n_subhalos)
            
            # Check coordinates are within box limits
            @test all(params.Lmin .<= galaxies.x .<= params.Lmax)
            @test all(params.Lmin .<= galaxies.y .<= params.Lmax)
            @test all(params.Lmin .<= galaxies.z .<= params.Lmax)
            
            # Check RSD effect
            if params.rsd
                # With RSD, z values should be different from original halos/subhalos
                matching_z = 0
                for i in 1:min(100, galaxies.count)
                    if any(isapprox.(galaxies.z[i], h_z, atol=1e-5)) || 
                       any(isapprox.(galaxies.z[i], s_z, atol=1e-5))
                        matching_z += 1
                    end
                end
                # Most z positions should be modified due to RSD
                @test matching_z < 50
            end
        end
    end
    
    @testset "Reproducibility with same seed" begin
        # Set a specific seed
        Random.seed!(42)
        
        # Use the first parameter set
        params = test_params[1]
        
        # Run the model twice with the same seed
        Random.seed!(42)
        galaxies1 = GalPop.populate_galaxies(halos, subhalos, params)
        
        Random.seed!(42)
        galaxies2 = GalPop.populate_galaxies(halos, subhalos, params)
        
        # Results should be identical
        @test galaxies1.count == galaxies2.count
        @test all(galaxies1.x .== galaxies2.x)
        @test all(galaxies1.y .== galaxies2.y)
        @test all(galaxies1.z .== galaxies2.z)
    end
    
    @testset "Different results with different parameters" begin
        # Use both parameter sets
        Random.seed!(42)
        galaxies1 = GalPop.populate_galaxies(halos, subhalos, test_params[1])
        
        Random.seed!(42)
        galaxies2 = GalPop.populate_galaxies(halos, subhalos, test_params[2])
        
        # Results should be different
        @test !(galaxies1.count == galaxies2.count && 
                all(galaxies1.x .== galaxies2.x) && 
                all(galaxies1.y .== galaxies2.y) && 
                all(galaxies1.z .== galaxies2.z))
    end
end