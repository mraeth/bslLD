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
        bslLD.use_cpu!()
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.1, 0.1], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        rho = bslLD.compute_density(f, grid)
        @test length(rho.data) == length(grid.xaxes[1])

        scalar = bslLD.ScalarField(reshape(collect(1.0:4.0), 2, 2))
        vector = bslLD.VectorField([collect(1.0:4.0), collect(5.0:8.0)])
        @test scalar.data isa Array
        @test all(component -> component isa bslLD.ScalarField, vector.data)
        @test all(component -> component.data isa Array, vector.data)
    end
end
