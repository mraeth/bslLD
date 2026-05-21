using Test
using bslLD

function zero_current_field(grid)
    dims = Tuple(length.(grid.xaxes))
    return bslLD.VectorField([zeros(dims) for _ in 1:3])
end

@testset "Moments" begin
    grid = bslLD.Grid([0.0], [2pi], [16], 0.05, 1, 1)
    rho = bslLD.ScalarField(sin.(grid.xaxes[1]))
    current = zero_current_field(grid)

    moments_rho = bslLD.Moments(rho)
    @test moments_rho.rho === rho
    @test moments_rho.J === nothing

    moments_full = bslLD.Moments(rho, current)
    @test moments_full.rho === rho
    @test moments_full.J === current

    bad_current = bslLD.VectorField([zeros(8) for _ in 1:3])
    @test_throws DimensionMismatch bslLD.Moments(rho, bad_current)
end

@testset "Field Solvers" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    @testset "Poisson 1D" begin
        grid = bslLD.Grid([0.0], [2pi], [32], 0.05, 1, 1, 2.5, 2)
        x = grid.xaxes[1]
        rho = bslLD.ScalarField(-sin.(x))
        solution = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonFieldSolver())

        @test length(solution.E) == 3
        @test length(solution.B) == 3
        @test length(solution.B0) == 3
        @test maximum(abs.(solution.E[1].data .+ cos.(x))) < 1e-10
        @test maximum(abs.(solution.E[2].data)) < 1e-10
        @test maximum(abs.(solution.E[3].data)) < 1e-10
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
        @test maximum(abs.(solution.B0[1].data)) < 1e-12
        @test maximum(abs.(solution.B0[2].data .- fill(2.5, length(x)))) < 1e-12
        @test maximum(abs.(solution.B0[3].data)) < 1e-12
    end

    @testset "Poisson 2D" begin
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [24, 20], 0.05, 1, 2, 1.25, 3)
        x = grid.xaxes[1]
        y = grid.xaxes[2]
        rho = bslLD.ScalarField([-sin(xi) + cos(yi) for xi in x, yi in y])
        solution = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonFieldSolver())

        expected_ex = [-cos(xi) for xi in x, yi in y]
        expected_ey = [-sin(yi) for xi in x, yi in y]
        @test maximum(abs.(solution.E[1].data .- expected_ex)) < 1e-10
        @test maximum(abs.(solution.E[2].data .- expected_ey)) < 1e-10
        @test maximum(abs.(solution.E[3].data)) < 1e-10
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
        @test maximum(abs.(solution.B0[1].data)) < 1e-12
        @test maximum(abs.(solution.B0[2].data)) < 1e-12
        @test maximum(abs.(solution.B0[3].data .- fill(1.25, length(x), length(y)))) < 1e-12
    end

    @testset "Poisson 3D" begin
        grid = bslLD.Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], [16, 12, 10], 0.05, 1, 3)
        x = grid.xaxes[1]
        y = grid.xaxes[2]
        z = grid.xaxes[3]
        rho = bslLD.ScalarField([
            -sin(xi) + cos(yi) - sin(zi) for xi in x, yi in y, zi in z
        ])
        solution = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonFieldSolver())

        expected_ex = [-cos(xi) for xi in x, yi in y, zi in z]
        expected_ey = [-sin(yi) for xi in x, yi in y, zi in z]
        expected_ez = [-cos(zi) for xi in x, yi in y, zi in z]
        @test maximum(abs.(solution.E[1].data .- expected_ex)) < 1e-9
        @test maximum(abs.(solution.E[2].data .- expected_ey)) < 1e-9
        @test maximum(abs.(solution.E[3].data .- expected_ez)) < 1e-9
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
    end

    @testset "Adiabatic" begin
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [24, 20], 0.05, 1, 2, 0.75, 1)
        x = grid.xaxes[1]
        y = grid.xaxes[2]
        rho = bslLD.ScalarField([sin(xi) + cos(2yi) for xi in x, yi in y])
        current = zero_current_field(grid)
        solution = bslLD.solve_fields(bslLD.Moments(rho, current), grid, bslLD.AdiabaticFieldSolver())

        expected_ex = [-cos(xi) for xi in x, yi in y]
        expected_ey = [2sin(2yi) for xi in x, yi in y]
        @test maximum(abs.(solution.E[1].data .- expected_ex)) < 1e-10
        @test maximum(abs.(solution.E[2].data .- expected_ey)) < 1e-10
        @test maximum(abs.(solution.E[3].data)) < 1e-10
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
        @test maximum(abs.(solution.B0[1].data .- fill(0.75, length(x), length(y)))) < 1e-12
        @test maximum(abs.(solution.B0[2].data)) < 1e-12
        @test maximum(abs.(solution.B0[3].data)) < 1e-12
    end
end
