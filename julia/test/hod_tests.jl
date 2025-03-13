# File: julia/test/hod_tests.jl

using Test
using Random
using Statistics
using GalPop
using Random

Random.rand() = 0.5  # Forces rand() to always return 0.5
Random.randn() = 0.5 # Forces randn() to always return 0.5

@testset "HOD Module Tests" begin
    
    @testset "calculate_p_cen function" begin
        
        test_mass = 10^13
        Mcut = test_mass
        sigma = 1/sqrt(2)
        p_value = GalPop.calculate_p_cen(test_mass, Mcut, sigma)
        @test isapprox(p_value, 0.5, atol = 1e-4)


        test_mass = 10^13
        Mcut = 10^14
        sigma = 1/sqrt(2)
        p_value = GalPop.calculate_p_cen(test_mass, Mcut, sigma)
        @test isapprox(p_value, 0.1573/2.0, atol = 1e-4)
    end
    
    @testset "calculate_n_sat function" begin

        Mh = 10^13
        Mcut = 10^14
        M1 = 10^14
        kappa = 1
        alpha = 1
        n_sat = GalPop.calculate_n_sat(Mh, Mcut, M1, alpha, kappa, 1.0)
        @test isapprox(n_sat, 0.0, atol = 1e-4)
        
        Mh = 2*10^13
        Mcut = 10^14
        M1 = 10^12
        kappa = 0.1
        alpha = 2
        n_sat = GalPop.calculate_n_sat(Mh, Mcut, M1, alpha, kappa, 1.0)
        @test isapprox(n_sat, 100, atol = 1e-4)
    end

    @testset "populate_galaxies" begin

        halos = (
            mass = [10^11, 10^12, 10^14, 10^15],
            x = [1.0, 2.0, 3.0, 4.0],
            y = [1.0, 2.0, 3.0, 4.0],
            z = [999.0, 999.0, 0.0, -999.0],
            velocity = [2.0, -1.0, 0.0, -2.0],
            sigma = [1.0, 1.0, 1.0, 1.0]
            )
        
        subhalos = (
            mass = [10^11, 10^12, 10^14, 10^15],
            host_velocity = [1.0, 2.0, 3.0, 4.0],
            n_particles = [10, 10, 10, 10],
            x = [1.0, 2.0, 3.0, 4.0],
            y = [1.0, 2.0, 3.0, 4.0],
            z = [1.0, 2.0, 3.0, 4.0],
            velocity = [-2.0, -1.0, 1.0, 2.0]
        )

        hod_params = (
            lnMcut = 13.0,
            sigma = 0.1,
            lnM1 = 13.0,
            kappa = 1,
            alpha = 1.0,
            alpha_c = 1.0,
            alpha_s = 2.0,
            rsd = true,
            Lmin = -1000.0,
            Lmax = 1000.0
        )

        x_out = [3.0, 4.0, 3.0, 4.0]
        y_out = [3.0, 4.0, 3.0, 4.0]
        z_out = [0.5, 999.5, 2.0, 4.0]
        count_out = 4
        result = populate_galaxies(halos, subhalos, hod_params)
        @test isapprox(x_out, result.x, atol = 1e-4)
        @test isapprox(y_out, result.y, atol = 1e-4)
        @test isapprox(z_out, result.z, atol = 1e-4)
        @test isapprox(count_out, result.count, atol = 1e-4)

        hod_params = (
            lnMcut = 13.0,
            sigma = 0.1,
            lnM1 = 13.0,
            kappa = 1,
            alpha = 1.0,
            alpha_c = 1.0,
            alpha_s = 2.0,
            rsd = false,
            Lmin = -1000.0,
            Lmax = 1000.0
        )

        z_out = [999.0, 999.0, 3.0, 4.0]
        result = populate_galaxies(halos, subhalos, hod_params)
        @test isapprox(z_out, result.z, atol = 1e-4)



    end

end