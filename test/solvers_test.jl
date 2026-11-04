using Test
using bslLD
using LinearAlgebra

function zero_current_field(grid)
    dims = Tuple(length.(grid.xaxes))
    return bslLD.VectorField([zeros(dims) for _ = 1:3])
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

    bad_current = bslLD.VectorField([zeros(8) for _ = 1:3])
    @test_throws DimensionMismatch bslLD.Moments(rho, bad_current)

    pi_diff = zero_current_field(grid)
    moments_em = bslLD.Moments(rho, current, pi_diff)
    @test moments_em.J === current
    @test moments_em.Pi_diff === pi_diff
end

@testset "Field Solvers" begin
    bslLD.set_execution_space!(exec = bslLD.backend())

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
        rho =
            bslLD.ScalarField([-sin(xi) + cos(yi) - sin(zi) for xi in x, yi in y, zi in z])
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
        solution =
            bslLD.solve_fields(bslLD.Moments(rho, current), grid, bslLD.AdiabaticSolver())
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

    @testset "Adiabatic with a mirror axis" begin
        # x: mirror grid 0:dx:Lx (walls on the end nodes), y: periodic
        Lx = 2.0
        function mirror_E(Nx, order)
            dx = Lx / (Nx - 1)
            grid = bslLD.Grid([0.0, 0.0], [Lx + dx, 2pi], [Nx, 12], 2)
            x, y = grid.xaxes
            # cos(pi x/Lx) is even about both walls, as the density of a mirrored distribution
            rho = bslLD.ScalarField([cos(pi * xi / Lx) + sin(yi) for xi in x, yi in y])
            solver = bslLD.AdiabaticSolver((bslLD.Mirror(), bslLD.Periodic()); order)
            sol = bslLD.solve_fields(
                bslLD.Moments(rho, zero_current_field(grid)),
                grid,
                solver,
            )
            ex = [pi / Lx * sin(pi * xi / Lx) for xi in x, yi in y]
            ey = [-cos(yi) for xi in x, yi in y]
            return sol.E, ex, ey
        end

        for order in (2, 4, 8)
            errs = map((17, 33)) do Nx
                E, ex, ey = mirror_E(Nx, order)
                @test maximum(abs.(E[2].data .- ey)) < 1e-12    # periodic axis stays spectral
                @test all(iszero, E[1].data[[1, end], :])       # E_x = 0 on both walls
                maximum(abs.(E[1].data .- ex))
            end
            @test log2(errs[1] / errs[2]) > order - 0.2          # convergence at the stencil order
        end
        @test bslLD.AdiabaticSolver().mirror_op === bslLD.FD(8, bslLD.Mirror())
        @test_throws ArgumentError bslLD.AdiabaticSolver(order = 3)
        @test_throws ArgumentError bslLD.AdiabaticSolver((bslLD.Fourier(),))
    end
end

@testset "EMSolverDKPol" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i = 0.5
    mu = 4.0
    dt = 0.1
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

        @test_throws ArgumentError bslLD.solve_fields!(
            sol,
            bslLD.Moments(rho),
            grid,
            solver,
            dt,
        )
        @test_throws ArgumentError bslLD.solve_fields!(
            sol,
            bslLD.Moments(rho, J),
            grid,
            solver,
            dt,
        )
    end
end

@testset "EMSolverDKNoPol" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i = 0.5
    mu = 4.0
    dt = 0.1
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
        @test maximum(abs.(sol.Enew[2].data .- J1_val)) < 1e-12
        @test maximum(abs.(sol.Enew[3].data)) < 1e-12
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

        @test_throws ArgumentError bslLD.solve_fields!(
            sol,
            bslLD.Moments(rho),
            grid,
            solver,
            dt,
        )
        @test_throws ArgumentError bslLD.solve_fields!(
            sol,
            bslLD.Moments(rho, J),
            grid,
            solver,
            dt,
        )
    end
end

@testset "EMSolverDKNoPol Faraday identity (random, spectral accuracy)" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i, mu = 0.5, 4.0
    dt = 0.3
    solver = bslLD.EMSolverDKNoPol(beta_i, mu)

    Nx, Ny = 32, 32
    grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)

    E_init = bslLD.VectorField([bslLD.ScalarField(randn(Nx, Ny)) for _ = 1:3])
    B_init = bslLD.VectorField([bslLD.ScalarField(randn(Nx, Ny)) for _ = 1:3])
    B0_data = [copy(B_init[d].data) for d = 1:3]

    curlB = bslLD.curl(B_init, grid)
    c_amp = 2.0 / beta_i
    J_perp = bslLD.VectorField([
        bslLD.ScalarField(c_amp .* curlB[1].data),
        bslLD.ScalarField(c_amp .* curlB[2].data),
        bslLD.ScalarField(zeros(Nx, Ny)),
    ])
    Pi_diff = bslLD.VectorField([bslLD.ScalarField(randn(Nx, Ny)) for _ = 1:3])

    moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_perp, Pi_diff)
    sol = bslLD.FieldSolution(E_init, B_init)
    bslLD.solve_fields!(sol, moments, grid, solver, dt)

    # Faraday: B^n = B^{n+1} + dt * curl(E^{n+1}) — exact to machine precision
    curlE_new = bslLD.curl(sol.Enew, grid)
    for d = 1:3
        B_reconstructed = sol.B[d].data .+ dt .* curlE_new[d].data
        @test maximum(abs.(B_reconstructed .- B0_data[d])) < 1e-10
    end
end


@testset "EMSolverDKNoPol perpendicular Ampere algebraic identity" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i, mu = 0.5, 4.0
    dt = 0.3
    solver = bslLD.EMSolverDKNoPol(beta_i, mu)
    Nx, Ny = 32, 32
    grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
    E_init = bslLD.VectorField([bslLD.ScalarField(randn(Nx, Ny)) for _ = 1:3])
    B_init = bslLD.VectorField([bslLD.ScalarField(randn(Nx, Ny)) for _ = 1:3])
    B_old = [copy(B_init[d].data) for d = 1:3]
    J_perp = bslLD.VectorField([
        bslLD.ScalarField(randn(Nx, Ny)),
        bslLD.ScalarField(randn(Nx, Ny)),
        bslLD.ScalarField(zeros(Nx, Ny)),
    ])
    Pi_diff = bslLD.VectorField([bslLD.ScalarField(randn(Nx, Ny)) for _ = 1:3])
    moments = bslLD.Moments(bslLD.empty_scalarfield(grid), J_perp, Pi_diff)
    sol = bslLD.FieldSolution(E_init, B_init)
    bslLD.solve_fields!(sol, moments, grid, solver, dt)

    # Algebraic perp Ampere at t^{n+1}: E^{n+1}_⊥ = [c·curl(B^{n+1})_⊥ − J^n] × ẑ
    curlB_new = bslLD.curl(sol.B, grid)
    c_amp = 2.0 / beta_i

    Ex_expected = c_amp .* curlB_new[2].data .- J_perp[2].data
    Ey_expected = .-(c_amp .* curlB_new[1].data .- J_perp[1].data)

    @test maximum(abs.(sol.Enew[1].data .- Ex_expected)) < 1e-8
    @test maximum(abs.(sol.Enew[2].data .- Ey_expected)) < 1e-8
end

@testset "solve_fields_midpoint! (EMSolverDKNoPol)" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i, mu, dt = 0.5, 4.0, 0.2
    solver = bslLD.EMSolverDKNoPol(beta_i, mu)
    Nx, Ny = 16, 14
    grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)

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
    B_before = [copy(B[d].data) for d = 1:3]
    sol = bslLD.FieldSolution(E, B)

    bslLD.solve_fields_midpoint!(sol, moments, grid, solver, dt)

    # B must not be mutated by the midpoint solve — caller applies Faraday separately.
    for d = 1:3
        @test sol.B[d].data == B_before[d]
    end

    # Same 3x3 system as a full solve_fields! call at dt/2 (with no Faraday update).
    sol_ref =
        bslLD.FieldSolution(bslLD.zero_vectorfield3(grid), bslLD.zero_vectorfield3(grid))
    bslLD.solve_fields!(sol_ref, moments, grid, solver, dt / 2)

    for d = 1:3
        @test maximum(abs.(sol.Enew[d].data .- sol_ref.Enew[d].data)) < 1e-12
    end
end

@testset "solve_fields_damped_midpoint! (EMSolverDKNoPol)" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i, mu, dt, theta = 0.5, 4.0, 0.2, 0.55
    solver = bslLD.EMSolverDKNoPol(beta_i, mu)
    Nx, Ny = 16, 14
    grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)

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
    B_before = [copy(B[d].data) for d = 1:3]
    sol = bslLD.FieldSolution(E, B)

    bslLD.solve_fields_damped_midpoint!(sol, moments, grid, solver, dt, theta)

    for d = 1:3
        @test sol.B[d].data == B_before[d]
    end

    # Same 3x3 system as a full solve_fields! call at theta*dt (with no Faraday update).
    sol_ref =
        bslLD.FieldSolution(bslLD.zero_vectorfield3(grid), bslLD.zero_vectorfield3(grid))
    bslLD.solve_fields!(sol_ref, moments, grid, solver, theta * dt)

    for d = 1:3
        @test maximum(abs.(sol.Enew[d].data .- sol_ref.Enew[d].data)) < 1e-12
    end
end

@testset "apply_faraday!" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    Nx, Ny = 32, 32
    grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
    x, y = grid.xaxes
    dt = 0.3

    Ez_mode = [sin(xi) for xi in x, yi in y]
    E = bslLD.VectorField([
        bslLD.ScalarField(zeros(Nx, Ny)),
        bslLD.ScalarField(zeros(Nx, Ny)),
        bslLD.ScalarField(Ez_mode),
    ])
    B = bslLD.zero_vectorfield3(grid)

    bslLD.apply_faraday!(B, E, grid, dt)

    # curl(E)_y = -∂_x E_z = -cos(x)  ⇒  B_y -= dt * curl(E)_y = dt*cos(x)
    expected_By = [dt * cos(xi) for xi in x, yi in y]
    @test maximum(abs.(B[1].data)) < 1e-10
    @test maximum(abs.(B[2].data .- expected_By)) < 1e-10
    @test maximum(abs.(B[3].data)) < 1e-10
end

@testset "predict_midpoint_current / predict_stage_current" begin
    Nx, Ny = 8, 6
    grid = bslLD.Grid([0.0, 0.0], [2pi, 2pi], [Nx, Ny], 2, 1.0, 3)
    sp = bslLD.Distribution(grid, 0.0; m = 4.0, q = 2.0)
    eas = bslLD.electric_acceleration_scale(sp)

    n_a = bslLD.ScalarField(fill(1.5, Nx, Ny))
    J_a = bslLD.VectorField([bslLD.ScalarField(fill(v, Nx, Ny)) for v in (0.1, 0.2, 0.3)])
    E = bslLD.VectorField([bslLD.ScalarField(fill(v, Nx, Ny)) for v in (1.0, -2.0, 0.5)])
    dt, theta = 0.4, 0.7

    J_mid = bslLD.predict_midpoint_current(n_a, J_a, E, sp, grid, dt)
    τ_mid = dt / 2
    for d = 1:3
        expected = J_a[d].data .+ eas .* n_a.data .* τ_mid .* E[d].data
        @test maximum(abs.(J_mid[d].data .- expected)) < 1e-12
    end

    J_stage = bslLD.predict_stage_current(n_a, J_a, E, sp, grid, dt, theta)
    τ_stage = theta * dt
    for d = 1:3
        expected = J_a[d].data .+ eas .* n_a.data .* τ_stage .* E[d].data
        @test maximum(abs.(J_stage[d].data .- expected)) < 1e-12
    end
end

@testset "EMSolverDKPolDensity" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i, mu = 0.5, 0.01
    c = 2 / beta_i
    ω_cold(w, k) = sqrt(c * k^2 * (1 + w * mu) / (1 + w * mu + w * mu * c * k^2))
    grid1d(L, Nx) = bslLD.Grid([0.0, -1.0, -1.0], [L, 1.0, 1.0], [Nx, 1, 1], 1, 1.0, 3)
    zero3(Nx) = bslLD.VectorField([bslLD.ScalarField(zeros(Nx)) for _ = 1:3])

    @testset "Argument validation" begin
        solver = bslLD.EMSolverDKPolDensity(beta_i, mu)
        sol = bslLD.FieldSolution(zero3(8), zero3(8))
        rho = bslLD.ScalarField(ones(8))
        J = bslLD.VectorField([zeros(8), zeros(8)])
        g_par = bslLD.Grid([0.0, -1.0, -1.0], [2pi, 1.0, 1.0], [8, 1, 1], 1, 1.0, 1)
        @test_throws ArgumentError bslLD.solve_fields!(sol, bslLD.Moments(rho, J), g_par, solver, 0.1)
        @test_throws ArgumentError bslLD.solve_fields!(sol, bslLD.Moments(rho), grid1d(2pi, 8), solver, 0.1)
    end

    @testset "Quasineutral B_z, single mode" begin
        Nx, L, m, w = 32, 4pi, 3, 0.7
        grid = grid1d(L, Nx)
        x = collect(grid.xaxes[1])
        k = 2pi * m / L
        rho = bslLD.ScalarField(1 .+ 1e-3 .* cos.(k .* x))
        J = bslLD.VectorField([zeros(Nx), 2e-3 .* sin.(k .* x)])
        B = zero3(Nx)
        bslLD.quasineutral_B!(B, rho, J, grid, bslLD.EMSolverDKPolDensity(beta_i, mu; w))
        # (1 + wμck²) B = δn + ∂_x(wμ J_y)
        expected = (1e-3 .* cos.(k .* x) .+ w * mu * 2e-3 * k .* cos.(k .* x)) ./ (1 + w * mu * c * k^2)
        @test maximum(abs.(B[3].data .- expected)) < 1e-15
    end

    @testset "k = 0 balance" begin
        Nx = 8
        grid = grid1d(2pi, Nx)
        sol = bslLD.FieldSolution(zero3(Nx), zero3(Nx))
        moments = bslLD.Moments(bslLD.ScalarField(ones(Nx)), bslLD.VectorField([fill(0.3, Nx), fill(0.7, Nx)]))
        bslLD.solve_fields!(sol, moments, grid, bslLD.EMSolverDKPolDensity(beta_i, mu), 0.1)
        @test maximum(abs.(sol.Enew[1].data .+ 0.7)) < 1e-14
        @test maximum(abs.(sol.Enew[2].data .- 0.3)) < 1e-14
    end

    @testset "Quasineutral electrons, oblique single mode" begin
        N, Lx, Lz, w = 8, 4pi, 8pi, 0.7
        grid = bslLD.Grid([0.0, 0.0, -1.0, -1.0], [Lx, Lz, 1.0, 1.0], [N, N, 1, 1], 2, 1.0, 2)
        X = [x for x in Iterators.product(grid.xaxes...)]
        kx, kz = 2pi / Lx, 2pi / Lz
        ph = [kx * x + kz * z for (x, z) in X]
        solver = bslLD.EMSolverDKPolDensity(beta_i, mu; w)
        # Bdir = 2: b̂ = x2, right-handed perpendicular frame (a, b) = (x3, x1)
        ions = bslLD.ColdIonFluid(grid)
        ions.delta_n.data .= 1e-3 .* cos.(ph)
        ions.u[3].data .= 2e-3 .* sin.(ph)
        el = bslLD.Species(bslLD.ColdFluid(grid, bslLD.DriftFluid()); q = -1, m = mu)
        B = bslLD.VectorField([bslLD.ScalarField(zeros(N, N)) for _ = 1:3])
        B[2].data .= 3e-4 .* cos.(ph)
        bslLD.quasineutral_drift!((ions, el), B, grid, solver)
        # N_e = n_i − wμ[c k² B_∥ − b̂·(i k × J)], b̂·∇×J⊥ = ∂_x3 J_x1 − ∂_x1 J_x3 = −kx·2e-3 cos
        k2 = kx^2 + kz^2
        expected = (1e-3 .- w * mu * (c * k2 * 3e-4 + kx * 2e-3)) .* cos.(ph)
        @test maximum(abs.(el.delta_n.data .- expected)) < 1e-15
        @test bslLD.quasineutrality_residual((ions, el), B, grid, solver) < 1e-17
    end

    cold_electrons(grid) = bslLD.Species(bslLD.ColdFluid(grid, bslLD.DriftFluid()); q = -1, m = mu)

    # One-step map of the cold step on the Fourier mode k = (k⊥, k∥) (Bdir = 3 in 1D, Bdir = 2 in
    # 2D): 9 complex amplitudes (δn_i, u_i, B, δN_e, u_e∥).
    function step_map(kp, kz, dt; w = 1.0, N = 4, κ = 0.0)
        grid = iszero(kz) ? grid1d(2pi / kp, N) :
               bslLD.Grid([0.0, 0.0, -1.0, -1.0], [2pi / kp, 2pi / kz, 1.0, 1.0], [N, N, 1, 1], 2, 1.0, 2)
        dims = Tuple(length(ax) for ax in grid.xaxes)
        ph = [sum((kp, kz)[1:length(x)] .* x) for x in Iterators.product(grid.xaxes...)]
        z3() = bslLD.VectorField([bslLD.ScalarField(zeros(dims)) for _ = 1:3])
        solver = bslLD.EMSolverDKPolDensity(beta_i, mu; w)
        state(fl, el, sol) = (fl.delta_n, fl.u..., sol.B..., el.delta_n, el.u[grid.Bdir])
        amp(a) = sum(a .* exp.(-im .* ph)) / length(a)
        G = zeros(ComplexF64, 9, 9)
        for j = 1:9
            fl, el, sol = bslLD.ColdIonFluid(grid), cold_electrons(grid), bslLD.FieldSolution(z3(), z3())
            state(fl, el, sol)[j].data .= cos.(ph)
            bslLD.step_charge_form!((fl, el), sol, grid, solver, dt; damping_rate = κ)
            G[:, j] .= [amp(v.data) for v in state(fl, el, sol)] ./ 0.5
        end
        return G
    end

    # Continuum dispersion D(ω)·E = 0 in the frame (a, b, ∥), k = (k⊥, 0, k∥): cold ions, cold
    # electron fluid, longitudinal part of the polarisation current, Darwin Ampère with Faraday.
    function D_cont(ω, kp, kz, w)
        κ = [kp, 0.0, kz]
        K = [0 -κ[3] κ[2]; κ[3] 0 -κ[1]; -κ[2] κ[1] 0]
        R = ComplexF64[0 1 0; -1 0 0; 0 0 0]                     # R u = u × b̂
        σ = inv(-im * ω * I(3) - R) - R + Diagonal([0, 0, im / (ω * mu)]) -
            im * ω * w * mu * κ * [kp, 0.0, 0.0]' / sum(abs2, κ)
        return c * im * K * K / ω - σ
    end
    function root(ω, kp, kz, w)
        f(z) = det(D_cont(z, kp, kz, w))
        for _ = 1:50
            δ = 1e-7 * abs(ω)
            Δ = f(ω) / ((f(ω + δ) - f(ω - δ)) / 2δ)
            ω -= Δ
            abs(Δ) < 1e-13 * abs(ω) && break
        end
        return ω
    end
    # Step-map spectrum: physical frequencies (|λ| > 1/2, 0.05 < Re ω < π/(2dt), sorted) and the
    # largest growth rate (|λ| − 1)/dt outside λ ≈ 1. At λ = 1 (k∥ ≠ 0) the cold model has a
    # Jordan block (neutral parallel flow streams ballistically, δn_i = δN_e grows linearly), whose
    # eigenvalues split by √eps. λ = 0: projected quasineutrality defects; λ = −1: undamped
    # parallel-Ampère defects (the multiplier mode of the stage form).
    function step_frequencies(kp, kz, dt, w)
        λ = eigvals(step_map(kp, kz, dt; w))
        ω = im .* log.(λ) ./ dt
        phys = @. abs(λ) > 0.5 && 0.05 < real(ω) < pi / (2dt)
        γ = maximum((abs(l) - 1) / dt for (l, o) in zip(λ, ω) if abs(o * dt) > 1e-4)
        return sort(real.(ω[phys])), γ
    end

    @testset "Cold modes vs continuum (2nd order), k = ($kp, $kz), w = $w" for (kp, kz) in
        ((2.0, 0.0), (2.0, 0.5), (1.0, 1.0)), w in (1.0, 0.0)
        (ω1, γ1), (ω2, γ2) = step_frequencies(kp, kz, 0.02, w), step_frequencies(kp, kz, 0.01, w)
        @test length(ω1) == length(ω2) == (iszero(kz) ? 1 : 2)
        @test max(γ1, γ2) < 1e-7                                 # O(dt³) amplitude error only
        for (a, b) in zip(ω1, ω2)
            r = real(root(b, kp, kz, w))
            e1, e2 = abs(a / r - 1), abs(b / r - 1)
            @test e2 < 5e-4
            @test 3.5 < e1 / e2 < 4.5                            # second order
        end
        iszero(kz) && @test ω2[1] ≈ ω_cold(w, kp) rtol = 5e-4
    end

    # θ = 1/2 + κ dt: the parallel-Ampère defect (λ = −1 at θ = 1/2) decays like −(1 − θ)/θ, i.e.
    # at rate ≈ 4κ; physical branches keep second order and are damped at rate ≈ κ (ω dt)².
    @testset "Stage shift θ = 1/2 + κ dt, k = ($kp, $kz)" for (kp, kz) in ((2.0, 0.0), (2.0, 0.5))
        κ = 1.0
        errs = map((0.02, 0.01)) do dt
            θ = 0.5 + κ * dt
            λ = eigvals(step_map(kp, kz, dt; κ))
            λd = λ[argmin(abs.(λ .+ 1))]
            if iszero(kz)
                @test λd ≈ -(1 - θ) / θ rtol = 1e-8
            else
                @test abs(λd) < 1 - 3κ * dt
            end
            ω = im .* log.(λ) ./ dt
            phys = [o for (o, l) in zip(ω, λ) if abs(l) > 0.5 && 0.05 < real(o) < pi / (2dt)]
            for o in phys
                γ = -imag(o)
                @test 0 < γ < 2κ * (real(o) * dt)^2
            end
            [abs(real(o) / real(root(real(o), kp, kz, 1.0)) - 1) for o in sort(phys, by = real)]
        end
        @test all(3.5 .< errs[1] ./ errs[2] .< 4.5)              # second order
    end

    @testset "Quasineutrality holds over time, oblique random seed" begin
        grid = bslLD.Grid([0.0, 0.0, -1.0, -1.0], [8pi, 16pi, 1.0, 1.0], [16, 16, 1, 1], 2, 1.0, 2)
        solver = bslLD.EMSolverDKPolDensity(beta_i, mu)
        z3() = bslLD.VectorField([bslLD.ScalarField(zeros(16, 16)) for _ = 1:3])
        fl, el, sol = bslLD.ColdIonFluid(grid), cold_electrons(grid), bslLD.FieldSolution(z3(), z3())
        fl.delta_n.data .= 1e-6 .* sin.(collect(grid.xaxes[1]) ./ 4 .+ collect(grid.xaxes[2])' ./ 8 .+ 0.3) .+
                           1e-6 .* cos.(3 .* collect(grid.xaxes[1]) ./ 4 .- collect(grid.xaxes[2])' ./ 4)
        bslLD.quasineutral_drift!((fl, el), sol.B, grid, solver)
        res = 0.0
        for _ = 1:200
            bslLD.step_charge_form!((fl, el), sol, grid, solver, 0.02)
            res = max(res, bslLD.quasineutrality_residual((fl, el), sol.B, grid, solver))
        end
        @test res < 1e-14                                        # absolute; δn ~ 1e-6
    end

    @testset "Fast wave capped at ω_LH" begin
        ω, _ = step_frequencies(20.0, 0.0, 0.005, 1.0)
        @test abs(ω[1] / ω_cold(1.0, 20.0) - 1) < 1e-3
        @test ω[1] < 1 / sqrt(mu)
    end
end

@testset "Linear DK electrons (charge form)" begin
    bslLD.set_execution_space!(exec = bslLD.backend())
    beta_i, mu = 0.5, 0.01
    grid2(N) = bslLD.Grid([0.0, 0.0, -1.0, -1.0], [8pi, 16pi, 1.0, 1.0], [N, N, 1, 1], 2, 1.0, 2)
    z3(dims) = bslLD.VectorField([bslLD.ScalarField(zeros(dims)) for _ = 1:3])
    drift_fluid(grid) = bslLD.ColdFluid(grid, bslLD.DriftFluid())

    # Cold ions + electrons (drift fluid or DK, same δN = 0 and u∥) on one oblique mode.
    function setup(grid, model)
        ph = [2pi * (x / grid.max[1] + z / grid.max[2]) + 0.3 for (x, z) in Iterators.product(grid.xaxes...)]
        ions = bslLD.ColdIonFluid(grid)
        ions.delta_n.data .= 1e-3 .* cos.(ph)
        ions.u[3].data .= 1e-3 .* sin.(ph)
        el = bslLD.Species(model; q = -1, m = mu)
        u = 1e-3 .* cos.(ph .+ 1)
        if model isa bslLD.ColdFluid
            el.u[2].data .= u
        else                                       # δF = -(u/v_th) F0' carries J∥ = q u
            dF0 = bslLD._dk_maxwellian(model)[2]
            el.dist.data .= reshape(-u .* sqrt(mu), size(u)..., 1) .* reshape(dF0, 1, 1, :)
        end
        return ions, el
    end
    function run(grid, model, dt, T)
        solver = bslLD.EMSolverDKPolDensity(beta_i, mu)
        ions, el = setup(grid, model)
        sol = bslLD.FieldSolution(z3(size(ions.delta_n.data)), z3(size(ions.delta_n.data)))
        bslLD.quasineutral_drift!((ions, el), sol.B, grid, solver)
        res = 0.0
        for _ = 1:round(Int, T / dt)
            bslLD.step_charge_form!((ions, el), sol, grid, solver, dt)
            res = max(res, bslLD.quasineutrality_residual((ions, el), sol.B, grid, solver))
        end
        return (ni = copy(ions.delta_n.data), B = copy(sol.B[2].data), res = res)
    end
    diff(a, b) = maximum(abs.(a.ni .- b.ni)) + maximum(abs.(a.B .- b.B))

    @testset "Model, normalisation, trait" begin
        md = bslLD.LinearDK(grid2(8); T_e = 0.2, Nv = 32)
        sp = bslLD.Species(md; q = -1, m = mu)
        F0, dF0 = bslLD._dk_maxwellian(md)
        v, dv = collect(md.grid.vaxes[1]), bslLD._dk_dv(md)
        @test sum(F0) * dv ≈ 1 atol = 1e-14
        @test sum(v .* dF0) * dv ≈ -1 atol = 1e-14
        @test bslLD.response_class(sp) isa bslLD.Drift
        @test bslLD.response_class(bslLD.ColdIonFluid(grid2(8))) isa bslLD.Magnetised
        @test sp.dist === md.dist
        @test_throws ArgumentError bslLD.LinearDK(grid2(8); T_e = 0.0)
    end

    @testset "Stage response equals the push (Fourier streaming)" begin
        grid, dt, θ = grid2(8), 0.05, 0.5
        _, el = setup(grid, bslLD.LinearDK(grid2(8); T_e = 0.2, Nv = 32))
        B = z3((8, 8))
        B[2].data .= 2e-3 .* [sin(x / 4 + z / 8) for (x, z) in Iterators.product(grid.xaxes...)]
        bslLD.push_free!(el, grid, dt, 1; B)
        r = bslLD.stage_response(el, grid, dt, θ; B)
        @test r.exact
        E = z3((8, 8))
        for d = 1:3
            E[d].data .= 1e-3 * d .* [cos(x / 2 - z / 4 + d) for (x, z) in Iterators.product(grid.xaxes...)]
        end
        Ja = bslLD.moments_at(el, grid, 0)[2]
        bslLD.push_field!(el, E, grid, dt)
        Jb = bslLD.moments_at(el, grid, 0)[2]
        bslLD.push_free!(el, grid, dt, 2; B = bslLD.apply_faraday!(B, E, grid, dt))
        ρ1 = bslLD.moments_at(el, grid, 0)[1]
        a, b, pz = bslLD._cf_frame(grid)
        sw, _ = bslLD._cf_kviews(ρ1, grid)
        F(x) = bslLD._fft_c(x, sw)
        Eh = (F(E[a].data), F(E[b].data), F(E[pz].data))
        @test maximum(abs.(F(ρ1) .- bslLD._affine_at(r.rho, Eh...))) < 1e-15
        @test maximum(abs.(F(Ja[pz] .+ θ .* (Jb[pz] .- Ja[pz])) .- bslLD._affine_at(r.J[3], Eh...))) < 1e-15
    end

    @testset "Only δf streams: δF = F0 δB∥ is stationary" begin
        grid = grid2(8)
        md = bslLD.LinearDK(grid; T_e = 0.2, Nv = 32)
        el = bslLD.Species(md; q = -1, m = mu)
        B = z3((8, 8))
        B[2].data .= 1e-3 .* [cos(x / 4 + z / 8) for (x, z) in Iterators.product(grid.xaxes...)]
        el.dist.data .= B[2].data .* reshape(bslLD._dk_maxwellian(md)[1], 1, 1, :)
        F = copy(el.dist.data)
        bslLD.push_free!(el, grid, 0.1, 1; B)
        @test maximum(abs.(el.dist.data .- F)) < 1e-17
        @test_throws ArgumentError bslLD.push_free!(el, grid, 0.1, 1)
    end

    @testset "k∥ = 0: identical to the drift fluid" begin
        grid = bslLD.Grid([0.0, -1.0, -1.0], [4pi, 1.0, 1.0], [8, 1, 1], 1, 1.0, 3)
        solver = bslLD.EMSolverDKPolDensity(beta_i, mu)
        x = collect(grid.xaxes[1])
        out = map((drift_fluid(grid), bslLD.LinearDK(grid; T_e = 1.0))) do model
            ions = bslLD.ColdIonFluid(grid)
            ions.delta_n.data .= 1e-3 .* cos.(x ./ 2)
            ions.u[2].data .= 1e-3 .* sin.(x)
            el = bslLD.Species(model; q = -1, m = mu)
            sol = bslLD.FieldSolution(z3(8), z3(8))
            bslLD.quasineutral_drift!((ions, el), sol.B, grid, solver)
            for _ = 1:50
                bslLD.step_charge_form!((ions, el), sol, grid, solver, 0.02)
            end
            (copy(ions.delta_n.data), copy(sol.B[3].data), copy(sol.E[1].data))
        end
        @test all(maximum(abs.(a .- b)) < 1e-15 for (a, b) in zip(out...))
    end

    @testset "T_e → 0, second order in dt, QN, Lagrange streaming" begin
        grid = grid2(8)
        fl = run(grid, drift_fluid(grid), 0.02, 2.0)
        e3 = diff(run(grid, bslLD.LinearDK(grid; T_e = 1e-3), 0.02, 2.0), fl)
        e4 = diff(run(grid, bslLD.LinearDK(grid; T_e = 1e-4), 0.02, 2.0), fl)
        @test 9 < e3 / e4 < 11                                    # O(T_e)

        dk(dt) = run(grid, bslLD.LinearDK(grid; T_e = 0.1), dt, 1.0)
        ref = dk(0.0025)
        @test 3.5 < diff(dk(0.02), ref) / diff(dk(0.01), ref) < 4.5
        @test ref.res < 1e-11                                     # absolute; amplitudes 1e-3

        lag(N) = (g = grid2(N); diff(run(g, bslLD.LinearDK(g; T_e = 0.1, method = bslLD.Lagrange(6)), 0.02, 1.0),
                                     run(g, bslLD.LinearDK(g; T_e = 0.1), 0.02, 1.0)))
        @test lag(8) / lag(16) > 20                               # corrector: converges with the stencil
    end
end
