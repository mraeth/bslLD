using Test
using bslLD

@testset "bslLD" begin
    include("basics_test.jl")
    include("advection_test.jl")
    include("index_test.jl")
    include("differential_operators_test.jl")
    include("solvers_test.jl")
    include("maxwell_test.jl")
end
