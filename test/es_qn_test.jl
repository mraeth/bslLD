using Test
using bslLD
using LinearAlgebra

# Electrostatic quasineutral step (es_quasineutral.jl) with resistive electrons.
@testset "ElectrostaticQN" begin
    τ, κn = 12.9, -0.345

    # 2D grid (x, y), Bdir = 3, one velocity point (fluid species only).
    grid2(kx, ky; N = 4) = bslLD.Grid([0.0, 0.0, -1.0, -1.0],
                                      [iszero(kx) ? 2pi : 2pi / kx, iszero(ky) ? 2pi : 2pi / ky, 1.0, 1.0],
                                      [N, N, 1, 1], 2, 1.0, 3)

    # One-step map on the Fourier mode (kx, ky) with cold Lorentz-fluid ions (no ion drive):
    # 5 complex amplitudes (δn_i, u_x, u_y, δn_e, φ^m); φ^{m−1} enters the polarisation density.
    function step_map(kx, ky, dt; α, αz = α, μ = 1e-3, pol = true)
        grid = grid2(kx, ky)
        ph = [kx * x + ky * y for (x, y) in Iterators.product(grid.xaxes...)]
        z3() = bslLD.VectorField([bslLD.ScalarField(zeros(size(ph))) for _ = 1:3])
        amp(a) = sum(a .* exp.(-im .* ph)) / length(a)
        G = zeros(ComplexF64, 5, 5)
        for j = 1:5
            ions = bslLD.ColdIonFluid(grid)
            el = bslLD.Species(bslLD.ResistiveElectrons(grid; tau = τ, alpha = α, alpha_zonal = αz,
                                                        kappa_n = κn, polarization = pol); q = -1, m = μ)
            sol = bslLD.FieldSolution(z3(), z3())
            state() = (ions.delta_n, ions.u[1], ions.u[2], el.delta_n, sol.phi)
            state()[j].data .= cos.(ph)
            bslLD.step_es_qn!((ions, el), sol, grid, bslLD.ElectrostaticQN(), dt)
            G[:, j] .= [amp(v.data) for v in state()] ./ 0.5
        end
        return G
    end

    # Roots of k²/(ω²−1) = (−κn k_y + iα/τ)/(ω + iα) + μk² (cold ions, plan eq. 5), as a cubic.
    function roots5(kx, ky; α, μ = 1e-3, pol = true)
        k2 = kx^2 + ky^2
        a, m = -κn * ky + im * α / τ, pol ? μ * k2 : 0.0
        c = ComplexF64[-m, -(a + im * α * m), k2 + m, im * α * k2 + a + im * α * m]   # ω³ … ω⁰
        iszero(c[1]) && return eigvals(ComplexF64[-c[3]/c[2] -c[4]/c[2]; 1 0])
        return eigvals(ComplexF64[-c[2]/c[1] -c[3]/c[1] -c[4]/c[1]; 1 0 0; 0 1 0])
    end

    stepω(G, dt) = im .* log.(eigvals(G)) ./ dt

    @testset "cold-ion modes vs (5), second order: k = ($kx, $ky), α = $α" for (kx, ky) in
        ((0.0, 2.0), (1.0, 1.5)), α in (0.0, 3.0, 100.0)
        R = filter(z -> abs(z) < 20, roots5(kx, ky; α))
        @test !isempty(R)
        errs = map((0.02, 0.01)) do dt
            ω = stepω(step_map(kx, ky, dt; α), dt)
            γmax = maximum(imag, R; init = 0.0)
            @test maximum(abs.(eigvals(step_map(kx, ky, dt; α)))) < exp(1.05 * γmax * dt) + 1e-6
            maximum(r -> minimum(abs.(ω .- r)) / abs(r), R)
        end
        @test errs[2] < 2e-3
        @test 3.0 < errs[1] / errs[2] < 5.0
    end

    @testset "D_perp is α_k = α + D⊥k⊥²" begin
        kx, ky, dt, D = 1.0, 1.5, 0.02, 0.058
        grid = grid2(kx, ky)
        ph = [kx * x + ky * y for (x, y) in Iterators.product(grid.xaxes...)]
        z3() = bslLD.VectorField([bslLD.ScalarField(zeros(size(ph))) for _ = 1:3])
        function one(α, Dp)
            ions = bslLD.ColdIonFluid(grid)
            el = bslLD.Species(bslLD.ResistiveElectrons(grid; tau = τ, alpha = α, kappa_n = κn, D_perp = Dp); q = -1, m = 1e-3)
            sol = bslLD.FieldSolution(z3(), z3())
            ions.delta_n.data .= cos.(ph); el.delta_n.data .= 0.3 .* sin.(ph)
            bslLD.step_es_qn!((ions, el), sol, grid, bslLD.ElectrostaticQN(), dt)
            return sol.phi.data, el.delta_n.data
        end
        a, b = one(0.0, D), one(D * (kx^2 + ky^2), 0.0)
        @test maximum(abs.(a[1] .- b[1])) < 1e-13 * maximum(abs.(a[1]))
        @test maximum(abs.(a[2] .- b[2])) < 1e-13
    end

    @testset "stiff modes stay bounded: k_y = 0, α_zonal = $αz, physical μ" for αz in (0.0, 10.0)
        μ = 1 / 73350
        for dt in (0.05, 0.02), kx in (1.0, 8.0)
            λ = eigvals(step_map(kx, 0.0, dt; α = 10.0, αz, μ))
            @test maximum(abs, λ) < 1 + 1e-6
        end
    end

    @testset "α → ∞ gives Boltzmann electrons, α = 0 the E×B response" begin
        kx, ky, dt = 0.0, 2.0, 0.01
        # Boltzmann + cold ions: k²/(ω²−1) = 1/τ + μk²
        μ, k2 = 1e-3, 4.0
        ωB = sqrt(1 + k2 / (1 / τ + μ * k2))
        ω = stepω(step_map(kx, ky, dt; α = 1e6, μ), dt)
        @test minimum(abs.(ω .- ωB)) / ωB < 1e-3
        ω0 = stepω(step_map(kx, ky, dt; α = 0.0, μ), dt)
        for r in filter(z -> abs(z) < 20, roots5(kx, ky; α = 0.0, μ))
            @test minimum(abs.(ω0 .- r)) / abs(r) < 1e-3
        end
    end

    @testset "gradient drive: density moment and κT part" begin
        N, Nv, vmax = 8, 48, 6.0
        grid = bslLD.Grid([0.0, 0.0, -vmax, -vmax], [2pi, 2pi, vmax, vmax], [N, N, Nv, Nv], 2, 1.0, 3)
        ions = bslLD.Distribution(grid, 1e-14; initFuncx = x -> 1.0)
        nI = bslLD.compute_density(ions.dist, grid).data
        E = bslLD.VectorField([bslLD.ScalarField(zeros(N, N)) for _ = 1:3])
        E[2].data .= [sin(x) * cos(y) for x in grid.xaxes[1], y in grid.xaxes[2]]
        dt, n0 = 0.1, 1.3
        bslLD.add_gradient_drive!(ions, grid, dt, E; kappa_n = κn, n0)
        dn = bslLD.compute_density(ions.dist, grid).data .- nI
        @test maximum(abs.(dn .+ dt * n0 * κn .* E[2].data)) < 1e-10
        f0 = copy(ions.dist.data)
        bslLD.add_gradient_drive!(ions, grid, dt, E; kappa_n = 0.0, kappa_T = 0.7)
        @test maximum(abs.(bslLD.compute_density(ions.dist, grid).data .- nI .- dn)) < 1e-8   # grid moment of (v²/2 − 1)F_M
        @test maximum(abs.(ions.dist.data .- f0)) > 1e-3
        # add_kappaT!(κ) = add_gradient_drive!(kappa_T = −κ)
        g = copy(ions.dist.data)
        bslLD.add_kappaT!(ions, grid, dt, 0.4, E)
        h = copy(ions.dist.data); ions.dist.data .= g
        bslLD.add_gradient_drive!(ions, grid, dt, E; kappa_n = 0.0, kappa_T = -0.4)
        @test maximum(abs.(ions.dist.data .- h)) < 1e-14
    end

    @testset "E×B nonlinearity of ResistiveElectrons" begin
        grid = bslLD.Grid([0.0, 0.0, -1.0, -1.0], [2pi, 2pi, 1.0, 1.0], [16, 16, 1, 1], 2, 1.0, 3)
        X = [(x, y) for x in grid.xaxes[1], y in grid.xaxes[2]]
        φ = [sin(x) * cos(2y) for (x, y) in X]
        δn = [cos(3x) * sin(y) for (x, y) in X]
        # v_E = (−∂yφ, ∂xφ) = (2 sin x sin 2y, cos x cos 2y), ∇δn = (−3 sin 3x sin y, cos 3x cos y)
        exact = [2sin(x) * sin(2y) * (-3sin(3x) * sin(y)) + cos(x) * cos(2y) * cos(3x) * cos(y) for (x, y) in X]
        NL = bslLD._esqn_nonlinear(φ, δn, grid)
        sw = bslLD.spectral_workspace(φ, grid)
        @test maximum(abs.(bslLD.ifft_spatial!(similar(φ), NL, sw) .- exact)) < 1e-12
    end

    @testset "BGK step (add_bgk!)" begin
        N, Nv, vmax = 6, 40, 6.0
        grid = bslLD.Grid([0.0, 0.0, -vmax, -vmax], [2pi, 2pi, vmax, vmax], [N, N, Nv, Nv], 2, 1.0, 3)
        xs, ys, v1, v2 = grid.xaxes..., grid.vaxes...
        FM(v, T = 1) = exp(-v^2 / (2T)) / sqrt(2pi * T)
        sp = bslLD.Distribution(grid, 0.0)
        # density modulation + a zero-density, odd part
        sp.dist.data .= [(1 + 0.1cos(x) * sin(y)) * FM(a) * FM(b) + 0.05 * sin(x) * a * FM(a) * FM(b)
                         for x in xs, y in ys, a in v1, b in v2]
        n0 = bslLD.compute_density(sp.dist, grid).data
        f0 = copy(sp.dist.data)
        bslLD.add_bgk!(sp, grid, 0.0, 0.1)
        @test sp.dist.data == f0                                           # ν = 0: identity
        ν, dt = 0.7, 0.3
        bslLD.add_bgk!(sp, grid, ν, dt)
        @test maximum(abs.(bslLD.compute_density(sp.dist, grid).data .- n0)) < 1e-13   # density conserved
        dv = grid.delta[3] * grid.delta[4]
        Fh = [FM(a) * FM(b) for a in v1, b in v2]; Fh ./= sum(Fh) * dv
        target = [n0[i, j] * Fh[k, l] for i = 1:N, j = 1:N, k = 1:Nv+1, l = 1:Nv+1]
        @test maximum(abs.((sp.dist.data .- target) .- (f0 .- target) .* exp(-ν * dt))) < 1e-14
        # the target is a fixed point, also for T ≠ 1
        for T in (1.0, 0.26)
            FhT = [FM(a, T) * FM(b, T) for a in v1, b in v2]; FhT ./= sum(FhT) * dv
            sp.dist.data .= [n0[i, j] * FhT[k, l] for i = 1:N, j = 1:N, k = 1:Nv+1, l = 1:Nv+1]
            g = copy(sp.dist.data)
            bslLD.add_bgk!(sp, grid, 2.0, 1.0; T)
            @test maximum(abs.(sp.dist.data .- g)) < 1e-14
        end
    end
end
