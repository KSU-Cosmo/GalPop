# julia/test/runtests.jl
using Test
using GalPop

@testset "GalPop.jl" begin
    # A simple test that doesn't depend on actual functionality yet
    @test isdefined(GalPop, :populate_galaxies)
end