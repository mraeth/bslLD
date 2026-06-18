using Adapt

@testset "Basics" begin
    bslLD.set_execution_space!(exec = bslLD.backend())

    @testset "Grid" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 1)
        simTime = bslLD.SimulationTime(0.05, 5.0; nmax = 100)

        @test simTime.dt == 0.05
        @test simTime.final_T == 5.0
        @test simTime.nmax == 100
        @test length(grid.xaxes) == 1
        @test length(grid.vaxes) == 1
        @test length(grid.xaxes[1]) == 4
        @test length(grid.vaxes[1]) == 9
        @test grid.b0 == 1.0
        @test grid.max[1] == 10.0
        @test grid.min[2] == -6.0
        @test grid.delta[1] == 2.5
        @test grid.xaxes isa Tuple
        @test grid.vaxes isa Tuple

        bslLD.advance!(simTime)
        @test simTime.step == 1
        @test simTime.current_T == 0.05
    end

    @testset "Grid Adaptation" begin
        grid = bslLD.Grid([0.0, -2.0], [2pi, 2.0], [8, 4], 1)
        adapted = Adapt.adapt(Array, grid)

        @test adapted.delta == grid.delta
        @test adapted.xaxes == grid.xaxes
        @test adapted.vaxes == grid.vaxes
        @test adapted.b0 == grid.b0
        @test adapted.Bdir == grid.Bdir
    end

    @testset "Distribution" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 1)
        f = bslLD.Distribution(grid, 0.01)

        @test size(f.data) == (length(grid.xaxes[1]), length(grid.vaxes[1]))
        @test f.data isa Array
        @test f.m == 1.0
        @test f.q == 1.0

        ion = bslLD.Distribution(grid, 0.01; m = 4.0, q = 2.0)
        @test ion.m == 4.0
        @test ion.q == 2.0
        @test bslLD.thermal_velocity(ion) == 0.5
        @test bslLD.electric_acceleration_scale(ion) == 1.0
        @test_throws ArgumentError bslLD.Distribution(grid, 0.01; m = 0.0)
    end

    @testset "Fields" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 1)
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
