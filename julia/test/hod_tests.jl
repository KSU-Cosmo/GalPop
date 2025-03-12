# File: julia/test/hod_tests.jl

using Test
using Random
using Statistics
using GalPop

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
        @test isapprox(p_value, 0.1573, atol = 1e-4)
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
end