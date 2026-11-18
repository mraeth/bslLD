# Electrostatic charge form with an x1 profile (step_es_charge_form!) and the full-f DK species (FullDK).
using LinearAlgebra: eigvals, pinv

function es_setup(; Nx = 16, Ny = 4, Nv = 8, ε = 1e-8, Lx = 100.0, Ly = 100.0, a = 0.05Lx / 2π, μ = 1 / 3670)
    vm = 5.0
    grid = bslLD.Grid([0.0, 0.0, -vm, -vm, -vm], [Lx, Ly, vm, vm, vm], [Nx, Ny, Nv, Nv, Nv], 2, 1.0, 3)
    x, y, v = collect(grid.xaxes[1]), collect(grid.xaxes[2]), collect(grid.vaxes[1])
    M = exp.(-v .^ 2 ./ 2)
    M ./= sum(M) * grid.delta[3]
    N(X) = exp(a * sin(2π * X / Lx))
    f = [N(x[i] + v[b]) * M[a_] * M[b] * M[c] for i in eachindex(x), j in eachindex(y), a_ in eachindex(v),
         b in eachindex(v), c in eachindex(v)]
    sf = bslLD.ScalarField(f)
    ions = bslLD.Species(1.0, 1.0, bslLD.DistributionGridImpl{2,3,bslLD.Cart,typeof(sf)}(sf))
    ni = bslLD.compute_density(ions.dist, grid).data
    el = bslLD.Species(bslLD.FullDK(grid, ni .* (1 .+ ε .* cos.(2π .* y' ./ Ly)); T_e = 1.0, Nv = 8); q = -1.0, m = μ)
    z3() = bslLD.VectorField([bslLD.ScalarField(zeros(Nx, Ny)) for _ in 1:3])
    return (ions, el), grid, bslLD.FieldSolution(z3(), z3())
end

@testset "ES charge form with a profile" begin
    @testset "FullDK density and drift class" begin
        sp, grid, _ = es_setup(; ε = 0.0)
        ni = bslLD.compute_density(sp[1].dist, grid).data
        @test bslLD.moments_at(sp[2], grid, 0.0)[1] ≈ -ni rtol = 1e-12
        @test bslLD.is_drift(sp[2]) && !bslLD.is_drift(sp[1])
    end

    @testset "quasineutral φ" begin
        sp, grid, sol = es_setup()
        solver = bslLD.ESChargeForm()
        bslLD.es_quasineutral_phi!(sol, sp, grid, solver)
        g = bslLD._es_geometry(grid)
        n = bslLD._es_mean_x(bslLD._species_density(sp[2], grid, 0.0), g)
        ρ̂ = bslLD._es_fft(sum(bslLD.moments_at(s, grid, 0.0)[1] for s in sp), g)
        φ̂ = bslLD._es_fft(sol.phi.data, g)
        P = solver.w * sp[2].m * bslLD._es_perp(g.D, n, g.k[2][1])
        @test maximum(abs, ρ̂[:, 2] .+ P * φ̂[:, 2]) <= 1e-14   # densities O(1)
    end

    @testset "model responses match the trial push" begin
        sp, grid, sol = es_setup(; ε = 0.0, Nv = 16)    # the ion model needs Δv ≲ 0.6 (3e-3; 12% at Nv = 8)
        dt = 0.05
        g = bslLD._es_geometry(grid)
        saved = map(bslLD._save, sp)
        x, y = collect(grid.xaxes[1]), collect(grid.xaxes[2])
        φ = 1e-6 .* exp.(-((x .- 50) ./ 15) .^ 2) .* cos.(2π .* y' ./ 100)
        for (i, s) in enumerate(sp)
            trial(φ_) = bslLD._es_trial_rho((s,), (saved[i],), φ_, bslLD.es_field(φ_, grid, g), grid, dt, 0.0)
            Δρ = bslLD._es_fft(trial(φ) .- trial(zero(φ)), g)[:, 2]
            n = bslLD._es_mean_x(bslLD._species_density(s, grid, 0.0), g)
            A = bslLD._es_species_operator(s, n, g.D, g.k[2]..., dt, grid.b0, true)
            @test maximum(abs, Δρ .- A * bslLD._es_fft(φ, g)[:, 2]) <= (i == 1 ? 1e-2 : 1e-4) * maximum(abs, Δρ)
        end
    end

    @testset "electrons on a fixed background: θ = 1/2 step map is neutral" begin
        # Step map on φ̂(x, k_y,min) from unit density perturbations; with the x-Nyquist mode projected out by
        # a Galerkin restriction its eigenvalues are g = (1 − iωΔt/2)/(1 + iωΔt/2), |g| = 1 (was up to 1.46).
        Nx, Lx, μ, dt = 16, 100.0, 1 / 3670, 0.5
        a = 0.05Lx / 2π
        grid = bslLD.Grid([0.0, 0.0, -1.0], [Lx, 100.0, 1.0], [Nx, 4, 2], 2, 1.0, 3)
        x, y = collect(grid.xaxes[1]), collect(grid.xaxes[2])
        n0 = [exp(a * sin(2π * x[i] / Lx)) for i in 1:Nx, j in 1:4]
        solver = bslLD.ESChargeForm(; background = n0)
        g = bslLD._es_geometry(grid)
        cols = map(1:Nx) do i
            pert = [1 + 1e-6 * (k == i) * cos(2π * y[j] / 100) / n0[k, j] for k in 1:Nx, j in 1:4]
            el = bslLD.Species(bslLD.FullDK(grid, n0 .* pert; T_e = 1.0, Nv = 8); q = -1.0, m = μ)
            sol = bslLD.FieldSolution(bslLD.VectorField([bslLD.ScalarField(zeros(Nx, 4)) for _ in 1:3]),
                                      bslLD.VectorField([bslLD.ScalarField(zeros(Nx, 4)) for _ in 1:3]))
            bslLD.es_quasineutral_phi!(sol, (el,), grid, solver)
            φ0 = bslLD._es_fft(sol.phi.data, g)[:, 2]
            bslLD.step_es_charge_form!((el,), sol, grid, solver, dt)
            (φ0, bslLD._es_fft(sol.phi.data, g)[:, 2])
        end
        L0, L1 = reduce(hcat, first.(cols)), reduce(hcat, last.(cols))
        ev = eigvals(L1 * pinv(L0; rtol = 1e-8))
        @test maximum(abs, ev) <= 1 + 1e-8
        @test count(>(1e-3), abs.(ev)) == Nx - 1          # one zero: the projected x-Nyquist mode
    end

    @testset "stable far above the fast-wave limit; lagged E×B is not" begin
        for (impl, bounded) in ((true, true), (false, false))
            sp, grid, sol = es_setup()
            solver = bslLD.ESChargeForm(; implicit_drift = impl)
            bslLD.es_quasineutral_phi!(sol, sp, grid, solver)
            φ0 = maximum(abs, sol.phi.data)
            phase = 0.0
            for n in 1:10
                bslLD.step_es_charge_form!(sp, sol, grid, solver, 0.05; phase, damping_rate = 1.0)
                phase += 0.05
            end
            @test (maximum(abs, sol.phi.data) < φ0) == bounded
        end
    end

    @testset "variable coefficients: the stage equation holds at finite amplitude" begin
        # Electrons on a fixed background, n_e = n0 (1 + 0.3 cos(k_y y + 0.3) sin(2 k_x x)). With the Euler E×B
        # push ρ(φ) is affine, so one correction with the exact operator (predictor n(x, y)) solves the stage
        # equation ρ(φ^m) + P*(φ^m − (1−θ)φ^n)/θ = 0; the y-averaged operator leaves an O(1) defect.
        Nx, Ny, Lx, μ, dt, κ = 16, 8, 100.0, 1 / 3670, 0.05, 1.0
        θ = 0.5 + κ * dt
        grid = bslLD.Grid([0.0, 0.0, -1.0], [Lx, 100.0, 1.0], [Nx, Ny, 2], 2, 1.0, 3)
        x, y = collect(grid.xaxes[1]), collect(grid.xaxes[2])
        n0 = [exp(0.05Lx / 2π * sin(2π * x[i] / Lx)) for i in 1:Nx, j in 1:Ny]
        ne = n0 .* [1 + 0.3 * cos(2π * y[j] / 100 + 0.3) * sin(4π * x[i] / Lx) for i in 1:Nx, j in 1:Ny]
        g = bslLD._es_geometry(grid)
        defect = map((true, false)) do vc
            el = bslLD.Species(bslLD.FullDK(grid, ne; T_e = 1.0, Nv = 4); q = -1.0, m = μ)
            solver = bslLD.ESChargeForm(; background = n0, variable_coefficients = vc)
            z3() = bslLD.VectorField([bslLD.ScalarField(zeros(Nx, Ny)) for _ in 1:3])
            sol = bslLD.FieldSolution(z3(), z3())
            bslLD.es_quasineutral_phi!(sol, (el,), grid, solver)
            φn = copy(sol.phi.data)
            nstar = bslLD._species_density(el, grid, 0.0)
            φm = bslLD.step_es_charge_form!((el,), sol, grid, solver, dt; damping_rate = κ, corrections = 1)
            Pφ = μ .* bslLD._es_perp(nstar, (φm .- (1 - θ) .* φn) ./ θ, g)
            maximum(abs, bslLD._es_project(bslLD.moments_at(el, grid, 0.0)[1] .+ n0 .+ Pφ, g)) / maximum(abs, Pφ)
        end
        @test defect[1] < 1e-8
        @test defect[2] > 0.1
    end

    @testset "dealiased RK4 E×B push: the corrections converge at Δt ≫ the fast-wave limit" begin
        # The stage operator's E×B term is the linearisation of the dealiased push, P{Pn, Pφ}; with the
        # collocation term instead the defect stalls at ~3e-2 (Δt = 0.05 ≈ 70 × the fast-wave limit).
        Nx, Ny, Lx, μ, dt, κ = 16, 8, 100.0, 1 / 3670, 0.05, 1.0
        θ = 0.5 + κ * dt
        grid = bslLD.Grid([0.0, 0.0, -1.0], [Lx, 100.0, 1.0], [Nx, Ny, 2], 2, 1.0, 3)
        x, y = collect(grid.xaxes[1]), collect(grid.xaxes[2])
        n0 = [exp(0.05Lx / 2π * sin(2π * x[i] / Lx)) for i in 1:Nx, j in 1:Ny]
        ne = n0 .* [1 + 1e-4 * cos(2π * y[j] / 100 + 0.3) * sin(4π * x[i] / Lx) for i in 1:Nx, j in 1:Ny]
        g = bslLD._es_geometry(grid)
        el = bslLD.Species(bslLD.FullDK(grid, ne; T_e = 1.0, Nv = 4, exb = :rk4, dealias = true); q = -1.0, m = μ)
        solver = bslLD.ESChargeForm(; background = n0, variable_coefficients = true)
        z3() = bslLD.VectorField([bslLD.ScalarField(zeros(Nx, Ny)) for _ in 1:3])
        sol = bslLD.FieldSolution(z3(), z3())
        bslLD.es_quasineutral_phi!(sol, (el,), grid, solver)
        φn = copy(sol.phi.data)
        nstar = bslLD._species_density(el, grid, 0.0)
        φm = bslLD.step_es_charge_form!((el,), sol, grid, solver, dt; damping_rate = κ, corrections = 4)
        Pφ = μ .* bslLD._es_perp(nstar, (φm .- (1 - θ) .* φn) ./ θ, g)
        @test maximum(abs, bslLD._es_project(bslLD.moments_at(el, grid, 0.0)[1] .+ n0 .+ Pφ, g)) /
              maximum(abs, Pφ) < 1e-6
    end
end
