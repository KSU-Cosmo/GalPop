# julia/test/runtests.jl
using Pkg
# Ensure we're in the right environment
Pkg.activate(".")
# Develop the package in-place
using Test
using GalPop

@testset "GalPop.jl" begin
    # Basic module test
    @test isdefined(GalPop, :populate_galaxies)
    
    # Include the HOD tests
    include("hod_tests.jl")
end