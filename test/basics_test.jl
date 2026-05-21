@testset "Basics" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    @testset "Grid" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 0.05, 100, 1)

        @test grid.dt == 0.05
        @test length(grid.xaxes) == 1
        @test length(grid.vaxes) == 1
        @test length(grid.xaxes[1]) == 4
        @test length(grid.vaxes[1]) == 9
        @test first(grid.time) == 0.0
        @test last(grid.time) == 5.0
        @test collect(grid.itime) == collect(1:100)
        @test grid.b0 == 1.0
    end

    @testset "Distribution" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)

        @test size(f.data) == (length(grid.xaxes[1]), length(grid.vaxes[1]))
        @test f.data isa Array
    end

    @testset "Fields" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 0.05, 100, 1)
        f = bslLD.Distribution(grid, 0.01)
        rho = bslLD.compute_density(f, grid)

        @test size(rho.data) == (length(grid.xaxes[1]),)

        scalar = bslLD.ScalarField(reshape(collect(1.0:4.0), 2, 2))
        vector = bslLD.VectorField([collect(1.0:4.0), collect(5.0:8.0)])

        @test scalar.data isa Array
        @test all(component -> component isa bslLD.ScalarField, vector.data)
        @test all(component -> component.data isa Array, vector.data)
    end
end
