using Test
using bslLD

function zero_current_field(grid)
    dims = Tuple(length.(grid.xaxes))
    return bslLD.VectorField([zeros(dims) for _ in 1:3])
end

@testset "Moments" begin
    grid = bslLD.Grid([0.0], [2pi], [16], 1)
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

    pi_diff = zero_current_field(grid)
    moments_em = bslLD.Moments(rho, current, pi_diff)
    @test moments_em.J === current
    @test moments_em.Pi_diff === pi_diff
end

@testset "Field Solvers" begin
    bslLD.set_execution_space!(exec=bslLD.backend())

    @testset "Poisson 1D" begin
        grid = bslLD.Grid([0.0], [2pi], [32], 1, 2.5, 2)
        x = grid.xaxes[1]
        rho = bslLD.ScalarField(-sin.(x))
        solution = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonSolver())
        B0 = bslLD.background_field(grid)

        @test length(solution.E) == 3
        @test length(solution.B) == 3
        @test maximum(abs.(solution.E[1].data .+ cos.(x))) < 1e-10
        @test maximum(abs.(solution.E[2].data)) < 1e-10
        @test maximum(abs.(solution.E[3].data)) < 1e-10
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
        @test maximum(abs.(B0[1].data)) < 1e-12
        @test maximum(abs.(B0[2].data .- fill(2.5, length(x)))) < 1e-12
        @test maximum(abs.(B0[3].data)) < 1e-12
    end

    @testset "Poisson 2D" begin
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [24, 20], 2, 1.25, 3)
        x = grid.xaxes[1]
        y = grid.xaxes[2]
        rho = bslLD.ScalarField([-sin(xi) + cos(yi) for xi in x, yi in y])
        solution = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonSolver())
        B0 = bslLD.background_field(grid)

        expected_ex = [-cos(xi) for xi in x, yi in y]
        expected_ey = [-sin(yi) for xi in x, yi in y]
        @test maximum(abs.(solution.E[1].data .- expected_ex)) < 1e-10
        @test maximum(abs.(solution.E[2].data .- expected_ey)) < 1e-10
        @test maximum(abs.(solution.E[3].data)) < 1e-10
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
        @test maximum(abs.(B0[1].data)) < 1e-12
        @test maximum(abs.(B0[2].data)) < 1e-12
        @test maximum(abs.(B0[3].data .- fill(1.25, length(x), length(y)))) < 1e-12
    end

    @testset "Poisson 3D" begin
        grid = bslLD.Grid([0.0, 0.0, 0.0], [2pi, 2pi, 2pi], [16, 12, 10], 3)
        x = grid.xaxes[1]
        y = grid.xaxes[2]
        z = grid.xaxes[3]
        rho = bslLD.ScalarField([
            -sin(xi) + cos(yi) - sin(zi) for xi in x, yi in y, zi in z
        ])
        solution = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.PoissonSolver())

        expected_ex = [-cos(xi) for xi in x, yi in y, zi in z]
        expected_ey = [-sin(yi) for xi in x, yi in y, zi in z]
        expected_ez = [-cos(zi) for xi in x, yi in y, zi in z]
        @test maximum(abs.(solution.E[1].data .- expected_ex)) < 1e-9
        @test maximum(abs.(solution.E[2].data .- expected_ey)) < 1e-9
        @test maximum(abs.(solution.E[3].data .- expected_ez)) < 1e-9
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
    end

    @testset "Adiabatic" begin
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [24, 20], 2, 0.75, 1)
        x = grid.xaxes[1]
        y = grid.xaxes[2]
        rho = bslLD.ScalarField([sin(xi) + cos(2yi) for xi in x, yi in y])
        current = zero_current_field(grid)
        solution = bslLD.solve_fields(bslLD.Moments(rho, current), grid, bslLD.AdiabaticSolver())
        B0 = bslLD.background_field(grid)

        expected_ex = [-cos(xi) for xi in x, yi in y]
        expected_ey = [2sin(2yi) for xi in x, yi in y]
        @test maximum(abs.(solution.E[1].data .- expected_ex)) < 1e-10
        @test maximum(abs.(solution.E[2].data .- expected_ey)) < 1e-10
        @test maximum(abs.(solution.E[3].data)) < 1e-10
        @test all(component -> maximum(abs.(component.data)) < 1e-12, solution.B)
        @test maximum(abs.(B0[1].data .- fill(0.75, length(x), length(y)))) < 1e-12
        @test maximum(abs.(B0[2].data)) < 1e-12
        @test maximum(abs.(B0[3].data)) < 1e-12
    end
end

@testset "EMSolverDKPol" begin
    bslLD.set_execution_space!(exec=bslLD.backend())
    beta_i = 0.5
    mu     = 4.0
    dt     = 0.1
    solver = bslLD.EMSolverDKPol(beta_i, mu)

    @testset "Faraday step (2D)" begin
        Nx, Ny = 16, 14
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
        x, y = grid.xaxes
        # E = (sin x, cos y, 0) → curl(E)_z = cos x + sin y, others analytically known
        E = bslLD.VectorField([
            bslLD.ScalarField([sin(xi) for xi in x, yi in y]),
            bslLD.ScalarField([cos(yi) for xi in x, yi in y]),
            bslLD.ScalarField(zeros(Nx, Ny)),
        ])
        B = bslLD.zero_vectorfield3(grid)
        J_zero = bslLD.zero_vectorfield3(grid)
        Pi_zero = bslLD.zero_vectorfield3(grid)
        moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_zero, Pi_zero)
        sol = bslLD.FieldSolution(E, B)

        bslLD.solve_fields!(sol, moments, grid, solver, dt)

        # curl(E)_x = dEz/dy - dEy/dz = 0,  curl(E)_y = dEx/dz - dEz/dx = 0
        # curl(E)_z = dEy/dx - dEx/dy = 0 - (-sin y) = sin y   [dEy/dx=0, dEx/dy=-sin y wait]
        # E_x = sin(x): dEx/dy = 0; E_y = cos(y): dEy/dx = 0 → curl_z = 0
        # curl(E)_x = dEz/dy - dEy/dz = 0 - 0 = 0 (no z-dim)
        # curl(E)_y = dEx/dz - dEz/dx = 0 - 0 = 0 (no z-dim)
        # sol.B = 0 - dt*curl(E^n) = all zeros here (curl(E)=0 for this E)
        @test maximum(abs.(sol.B[1].data)) < 1e-10
        @test maximum(abs.(sol.B[2].data)) < 1e-10
        @test maximum(abs.(sol.B[3].data)) < 1e-10
    end

    @testset "E_perp 2x2 solve (2D)" begin
        Nx, Ny = 16, 14
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
        x, y = grid.xaxes

        # Zero B^{n+1} → curlB = 0; source from J_i_perp only
        E = bslLD.zero_vectorfield3(grid)
        B = bslLD.zero_vectorfield3(grid)

        # Constant J_i_perp = (J1, J2)
        J1_val, J2_val = 0.3, 0.7
        J_perp = bslLD.VectorField([
            bslLD.ScalarField(fill(J1_val, Nx, Ny)),
            bslLD.ScalarField(fill(J2_val, Nx, Ny)),
            bslLD.ScalarField(zeros(Nx, Ny)),
        ])
        Pi_zero = bslLD.zero_vectorfield3(grid)
        moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_perp, Pi_zero)
        sol = bslLD.FieldSolution(E, B)

        bslLD.solve_fields!(sol, moments, grid, solver, dt)

        # E^n = 0, curlB = 0 → RHS1 = -α*J1, RHS2 = -α*J2
        α = dt / mu
        RHS1 = -α * J1_val
        RHS2 = -α * J2_val
        denom = 1 + α^2
        expected_E1 = (RHS1 + α * RHS2) / denom
        expected_E2 = (RHS2 - α * RHS1) / denom

        @test maximum(abs.(sol.Enew[1].data .- expected_E1)) < 1e-12
        @test maximum(abs.(sol.Enew[2].data .- expected_E2)) < 1e-12
    end

    @testset "Helmholtz E_z solve (2D)" begin
        Nx, Ny = 32, 32
        Lx, Ly = 2pi, 2pi
        grid = bslLD.Grid([0.0, 0.0], [Lx, Ly], [Nx, Ny], 2, 1.0, 3)
        x, y = grid.xaxes

        # Pi_diff_z = (cos x, 0) → ∇·Pi_diff_z = -sin x
        # E_perp = 0 → ∇_⊥·E_⊥ = 0, no ∂_z term (2D grid)
        # RHS = beta_i/2 * (-sin x)
        # Helmholtz: (-k_x² - k_y² - beta_i/2*(1+1/mu)) E_z = RHS
        # For mode (kx=1, ky=0): E_z = (beta_i/2*sin x) / (1 + beta_i/2*(1+1/mu))
        coeff = (beta_i / 2) / (1 + beta_i / 2 * (1 + 1/mu))
        expected_Ez = [coeff * sin(xi) for xi in x, yi in y]

        Pi_diff = bslLD.VectorField([
            bslLD.ScalarField([cos(xi) for xi in x, yi in y]),
            bslLD.ScalarField(zeros(Nx, Ny)),
            bslLD.ScalarField(zeros(Nx, Ny)),
        ])
        J_zero = bslLD.zero_vectorfield3(grid)
        moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_zero, Pi_diff)

        E = bslLD.zero_vectorfield3(grid)
        B = bslLD.zero_vectorfield3(grid)
        sol = bslLD.FieldSolution(E, B)

        bslLD.solve_fields!(sol, moments, grid, solver, dt)

        @test maximum(abs.(sol.Enew[3].data .- expected_Ez)) < 1e-10
    end

    @testset "Argument validation" begin
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [8, 8], 2, 1.0, 3)
        E = bslLD.zero_vectorfield3(grid)
        B = bslLD.zero_vectorfield3(grid)
        sol = bslLD.FieldSolution(E, B)
        rho = bslLD.empty_scalarfield(grid)
        J = bslLD.zero_vectorfield3(grid)
        Pi = bslLD.zero_vectorfield3(grid)

        @test_throws ArgumentError bslLD.solve_fields!(sol, bslLD.Moments(rho), grid, solver, dt)
        @test_throws ArgumentError bslLD.solve_fields!(sol, bslLD.Moments(rho, J), grid, solver, dt)

        grid_wrong_bdir = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [8, 8], 2, 1.0, 1)
        sol2 = bslLD.FieldSolution(bslLD.zero_vectorfield3(grid_wrong_bdir),
                                   bslLD.zero_vectorfield3(grid_wrong_bdir))
        @test_throws ArgumentError bslLD.solve_fields!(sol2, bslLD.Moments(rho, J, Pi), grid_wrong_bdir, solver, dt)
    end
end

@testset "EMSolverDKNoPol" begin
    bslLD.set_execution_space!(exec=bslLD.backend())
    beta_i = 0.5
    mu     = 4.0
    dt     = 0.1
    solver = bslLD.EMSolverDKNoPol(beta_i, mu)

    @testset "E_perp from constant J (2D)" begin
        Nx, Ny = 16, 14
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
        x, y = grid.xaxes

        J1_val, J2_val = 0.3, 0.7
        J_perp = bslLD.VectorField([
            bslLD.ScalarField(fill(J1_val, Nx, Ny)),
            bslLD.ScalarField(fill(J2_val, Nx, Ny)),
            bslLD.ScalarField(zeros(Nx, Ny)),
        ])
        Pi_zero = bslLD.zero_vectorfield3(grid)
        moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_perp, Pi_zero)
        E = bslLD.zero_vectorfield3(grid)
        B = bslLD.zero_vectorfield3(grid)
        sol = bslLD.FieldSolution(E, B)

        bslLD.solve_fields!(sol, moments, grid, solver, dt)

        # At k=0 with B=0: A = diag(1,1,-λ), b = R(-J1,-J2) = (-J2, J1, 0)
        # → E_x = -J2, E_y = J1, E_z = 0
        @test maximum(abs.(sol.Enew[1].data .- (-J2_val))) < 1e-12
        @test maximum(abs.(sol.Enew[2].data .- J1_val))    < 1e-12
        @test maximum(abs.(sol.Enew[3].data))               < 1e-12
    end

    @testset "E_z and B from Pi_diff (2D)" begin
        Nx, Ny = 32, 32
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
        x, y = grid.xaxes

        # Pi_diff[1] = cos(x) → div(Pi) = -sin(x) → E_z = (β/2)/(kx²+λ) sin(x)
        Pi_diff = bslLD.VectorField([
            bslLD.ScalarField([cos(xi) for xi in x, yi in y]),
            bslLD.ScalarField(zeros(Nx, Ny)),
            bslLD.ScalarField(zeros(Nx, Ny)),
        ])
        J_zero = bslLD.zero_vectorfield3(grid)
        moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_zero, Pi_diff)
        E = bslLD.zero_vectorfield3(grid)
        B = bslLD.zero_vectorfield3(grid)
        sol = bslLD.FieldSolution(E, B)

        bslLD.solve_fields!(sol, moments, grid, solver, dt)

        λ = (beta_i / 2) * (1 + 1 / mu)
        coeff = (beta_i / 2) / (1 + λ)     # kx=1 mode: denominator = kx² + λ = 1 + λ
        expected_Ez = [coeff * sin(xi) for xi in x, yi in y]
        @test maximum(abs.(sol.Enew[1].data)) < 1e-10
        @test maximum(abs.(sol.Enew[2].data)) < 1e-10
        @test maximum(abs.(sol.Enew[3].data .- expected_Ez)) < 1e-10

        # Faraday: curl(E)_y = -∂_x E_z = -coeff cos(x)
        # B_y^{n+1} = 0 - dt * (-coeff cos(x)) = dt*coeff*cos(x)
        expected_By = [dt * coeff * cos(xi) for xi in x, yi in y]
        @test maximum(abs.(sol.B[1].data)) < 1e-10
        @test maximum(abs.(sol.B[2].data .- expected_By)) < 1e-10
        @test maximum(abs.(sol.B[3].data)) < 1e-10
    end

    @testset "Argument validation" begin
        grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [8, 8], 2, 1.0, 3)
        E = bslLD.zero_vectorfield3(grid)
        B = bslLD.zero_vectorfield3(grid)
        sol = bslLD.FieldSolution(E, B)
        rho = bslLD.empty_scalarfield(grid)
        J = bslLD.zero_vectorfield3(grid)
        Pi = bslLD.zero_vectorfield3(grid)

        @test_throws ArgumentError bslLD.solve_fields!(sol, bslLD.Moments(rho), grid, solver, dt)
        @test_throws ArgumentError bslLD.solve_fields!(sol, bslLD.Moments(rho, J), grid, solver, dt)
    end
end
