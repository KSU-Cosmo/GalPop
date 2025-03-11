# julia/test/runtests.jl
using Test
using GalPop

@testset "GalPop.jl" begin
    # Basic module test
    @test isdefined(GalPop, :populate_galaxies)
    
    # Include the HOD tests
    include("hod_tests.jl")
end