# File: julia/test/hod_tests.jl

using Test
using Random
using Statistics
using GalPop
using Mocking
Mocking.activate()
using Random

rand_patch = @patch rand() = 0.5
randn_patch = @patch randn() = 0.5

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

        hod_params = [10^13, 0.1, 10^14-1.0, 1.0, 1.0, 2.0, 0.0, true, -1000.0, 1000.0]

        x_out = [3.0, 4.0, 3.0, 4.0]
        y_out = [3.0, 4.0, 3.0, 4.0]
        z_out = [0.5, 999.5, 2.0, −991.5]
        count_out = 4
        
        apply([rand_patch, randn_patch]) do
            begin
                x, y, z, count = populate_galaxies(halos, subhalos, hod_params)
            end
        end

        @test isapprox(x, x_out, atol = 1e-4)
        @test isapprox(y, y_out)
        @test isapprox(z, z_out)
        @test isapprox(count, count_out)
    end

end