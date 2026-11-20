# 3/2-padded E×B bracket (exb_bracket_padded!, exb_rk4!(…; dealias = :pad)) and its stage operator.
using LinearAlgebra: norm

@testset "3/2-padded E×B bracket" begin
    L1, L2, N1, N2 = 100.0, 60.0, 16, 12
    x = collect(range(0, L1; length = N1 + 1))[1:N1]
    y = collect(range(0, L2; length = N2 + 1))[1:N2]
    kn(N, L) = (k = collect(2π .* bslLD.fftfreq(N, N / L)); iseven(N) && (k[N÷2+1] = 0); k)
    k1, k2 = kn(N1, L1), kn(N2, L2)
    # fields with modes up to |m| < N/2 (no Nyquist): their product reaches 2N/2 − 2 > N/2, so collocation aliases
    modes(N) = 1:(N ÷ 2 - 1)
    function field(c, s)
        a = zeros(N1, N2)
        for (i, m1) in enumerate(modes(N1)), (j, m2) in enumerate(modes(N2))
            a .+= c * cos(i + s * j) / (1 + i + j) .* cos.(2π * m1 .* x ./ L1 .+ s * i .+ 2π * m2 .* y' ./ L2 .+ 0.7j)
        end
        return a
    end
    f, g = field(1.0, 0.3), field(0.5, 1.1)

    @testset "exact for the kept modes (4× oversampled reference)" begin
        # reference: the same derivatives on a 4× finer grid (no aliasing), product there, truncated to |m| < N/2
        M1, M2 = 4N1, 4N2
        up(a) = real(bslLD.ifft(bslLD._pad12(bslLD.fft(complex(a)), M1, M2))) .* (M1 * M2 / (N1 * N2))
        d(a, K) = real(bslLD.ifft(im .* K .* bslLD.fft(complex(a))))
        K1, K2 = reshape(k1, :, 1), reshape(k2, 1, :)
        h = up(d(f, K1)) .* up(d(g, K2)) .- up(d(g, K1)) .* up(d(f, K2))
        ref = real(bslLD.ifft(bslLD._trunc12(bslLD.fft(complex(h)), N1, N2) ./ (M1 * M2 / (N1 * N2))))
        out = bslLD.exb_bracket_padded!(similar(f), f, g, k1, k2)
        @test norm(out .- ref) / norm(ref) < 1e-12
        col = d(f, K1) .* d(g, K2) .- d(g, K1) .* d(f, K2)          # collocation: aliased
        @test norm(col .- ref) / norm(ref) > 1e-3
    end

    @testset "skew-symmetry and the 2/3 truncation on band-limited input" begin
        out = bslLD.exb_bracket_padded!(similar(f), f, g, k1, k2)
        @test abs(sum(f .* out)) / (norm(f) * norm(out)) < 1e-13
        @test abs(sum(out)) / norm(out) < 1e-13
        # inputs within 1/3 of the band: the product reaches 2N/3; collocation aliases only onto modes above N/3, so
        # after the 2/3 truncation of the output padding = collocation (the guarantee of the 2/3 rule)
        ff, gg = bslLD.exb_dealias!(copy(f)), bslLD.exb_dealias!(copy(g))
        K1, K2 = reshape(k1, :, 1), reshape(k2, 1, :)
        d(a, K) = real(bslLD.ifft(im .* K .* bslLD.fft(complex(a))))
        col = d(ff, K1) .* d(gg, K2) .- d(gg, K1) .* d(ff, K2)
        P(a) = bslLD.exb_dealias!(copy(a))
        @test norm(P(bslLD.exb_bracket_padded!(similar(ff), ff, gg, k1, k2)) .- P(col)) / norm(P(col)) < 1e-12
    end

    @testset "cached-buffer path = exb_bracket_padded! (with a velocity axis)" begin
        grid = bslLD.Grid([0.0, 0.0, -1.0], [L1, L2, 1.0], [N1, N2, 6], 2, 1.0, 3)
        fv = cat((c .* f for c in 1:7)...; dims = 3) .+ reshape(collect(1:7) .* 0.1, 1, 1, :) .* g
        P = bslLD._get_pad_plan(fv, grid)
        g1, g2 = bslLD._padded_gradient(g, k1, k2, P.M)
        a = bslLD._exb_bracket_padded!(similar(fv), fv, g1, g2, P)
        b = bslLD.exb_bracket_padded!(similar(fv), fv, g, k1, k2)
        @test norm(a .- b) / norm(b) < 1e-13
        bslLD.release_pad_buffers!(fv)
    end

    @testset "padded RK4 push conserves mass, with and without walls" begin
        μ = 1 / 3670
        for walls in (false, true)
            Nx = walls ? 17 : 16
            grid = bslLD.Grid([0.0, 0.0, -1.0], [walls ? Nx * 100 / (Nx - 1) : 100.0, 60.0, 1.0], [Nx, N2, 2], 2, 1.0, 3)
            xx, yy = collect(grid.xaxes[1]), collect(grid.xaxes[2])
            n0 = [exp(0.1 * cos(π * xx[i] / 100)) * (1 + 0.2 * cos(2π * 3 * yy[j] / 60)) for i in 1:Nx, j in 1:N2]
            φ = [0.05 * sin(2π * 2 * xx[i] / 100) * cos(2π * 4 * yy[j] / 60 + 0.4) for i in 1:Nx, j in 1:N2]
            el = bslLD.Species(bslLD.FullDK(grid, n0; T_e = 1.0, Nv = 8, exb = :rk4, dealias = :pad,
                                            x_boundary = walls ? :mirror : :periodic); q = -1.0, m = μ)
            md = getfield(el, :model)
            w = ones(Nx); walls && (w[[1, Nx]] .= 0.5)
            mass(F) = sum(w .* sum(F; dims = 2:3))
            m0, f0 = mass(md.dist.data), copy(md.dist.data)
            foreach(_ -> bslLD.push_drift!(el, bslLD.ScalarField(φ), grid, 20.0), 1:5)
            @test norm(md.dist.data .- f0) / norm(f0) > 1e-3
            @test abs(mass(md.dist.data) - m0) / m0 < 1e-13
        end
    end

    @testset "stage operator of the padded push: the corrections converge" begin
        # As the 2/3 test of es_charge_form_test.jl: with the stage operator the linearisation of the padded push,
        # the corrections converge at Δt ≫ the fast-wave limit; with walls as well.
        μ, dt, κ = 1 / 3670, 0.05, 1.0
        θ = 0.5 + κ * dt
        for walls in (false, true)
            Nx, Ny = walls ? 17 : 16, 8
            grid = bslLD.Grid([0.0, 0.0, -1.0], [walls ? Nx * 100 / (Nx - 1) : 100.0, 100.0, 1.0], [Nx, Ny, 2], 2, 1.0, 3)
            xx, yy = collect(grid.xaxes[1]), collect(grid.xaxes[2])
            n0 = [exp(0.05 * 100 / 2π * cos(π * xx[i] / 100)) for i in 1:Nx, j in 1:Ny]
            ne = n0 .* [1 + 1e-4 * cos(2π * yy[j] / 100 + 0.3) * sin(2π * xx[i] / 100) for i in 1:Nx, j in 1:Ny]
            g = bslLD._es_geometry(grid; x_boundary = walls ? :dirichlet : :periodic)
            el = bslLD.Species(bslLD.FullDK(grid, ne; T_e = 1.0, Nv = 4, exb = :rk4, dealias = :pad,
                                            x_boundary = walls ? :mirror : :periodic); q = -1.0, m = μ)
            solver = bslLD.ESChargeForm(; background = n0, variable_coefficients = true,
                                        x_boundary = walls ? :dirichlet : :periodic)
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
end
