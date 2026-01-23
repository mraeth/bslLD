using Test
using bslLD

@testset "bslLD.jl" begin
    @testset "Grid" begin
        # Test 1D1v grid creation
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.1, 0.1], 0.05, 100, 1)
        @test grid.dt == 0.05
        @test grid.nsteps == 100
        @test length(grid.xaxes) == 1
        @test length(grid.vaxes) == 1
    end

    @testset "Distribution" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.1, 0.1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        @test size(f.data, 1) == length(grid.xaxes[1])
        @test size(f.data, 2) == length(grid.vaxes[1])
    end

    @testset "Fields" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.1, 0.1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        rho = bslLD.compute_density(f, grid)
        @test length(rho.data) == length(grid.xaxes[1])
    end
end
