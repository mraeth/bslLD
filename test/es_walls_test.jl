# Walls along x1: ESChargeForm(x_boundary = :dirichlet) and FullDK(x_boundary = :mirror) on the Mirror() grid (walls on
# the first and last node), φ = 0 on the walls, quasineutrality on the interior nodes.
using LinearAlgebra: norm

function walls_setup(; N1 = 17, Ny = 8, L = 100.0)
    grid = bslLD.Grid([0.0, 0.0, -1.0], [N1 * L / (N1 - 1), 100.0, 1.0], [N1, Ny, 2], 2, 1.0, 3)
    x, y = collect(grid.xaxes[1]), collect(grid.xaxes[2])
    return grid, x, y, L
end

@testset "walls along x1" begin
    μ = 1 / 3670
    @testset "parity derivatives and the Dirichlet operator against a dense sine/cosine reference" begin
        grid, x, y, L = walls_setup()
        N = length(x)
        @test x[1] == 0 && x[end] ≈ L
        g = bslLD._es_geometry(grid; x_boundary = :dirichlet)
        err = maximum(1:N-2) do m
            s, c = sin.(m * π .* x ./ L), cos.(m * π .* x ./ L)
            max(norm(g.D * s .- (m * π / L) .* c), norm(g.De * c .+ (m * π / L) .* s)) / (m * π / L)
        end
        @test err < 1e-12
        # ∂x(n ∂x φ) on the interior nodes: sine coefficients of φ, cosine coefficients (DCT-I) of n φ'.
        n = 1 .+ 0.3 .* cos.(π .* x ./ L) .+ 0.1 .* cos.(3π .* x ./ L)
        I = 2:N-1
        S = [sin(m * π * x[j] / L) for j in I, m in 1:N-2]
        C = [cos(m * π * x[j] / L) for j in 1:N, m in 0:N-1]
        φ = zeros(N)
        φ[I] = randn(N - 2)
        a = S \ φ[I]
        dφ = [sum(a[m] * (m * π / L) * cos(m * π * x[j] / L) for m in 1:N-2) for j in 1:N]
        b = C \ (n .* dφ)
        ref = [sum(-b[m+1] * (m * π / L) * sin(m * π * x[j] / L) for m in 0:N-1) for j in I]
        op = bslLD._es_perp(g, n, 0.0) * φ
        @test norm(op[I] .- ref) / norm(ref) < 1e-11
    end

    @testset "quasineutral φ: zero on the walls, exact on the interior" begin
        grid, x, y, L = walls_setup()
        N, Ny = length(x), length(y)
        n0 = [exp(0.1 * cos(π * x[i] / L)) for i in 1:N, j in 1:Ny]
        ne = n0 .* [1 + 1e-4 * cos(2π * y[j] / 100 + 0.3) * sin(2π * x[i] / L) * (1 + 0.5cos(π * x[i] / L)) for i in 1:N, j in 1:Ny]
        for vc in (false, true)
            el = bslLD.Species(bslLD.FullDK(grid, ne; T_e = 1.0, Nv = 8, x_boundary = :mirror); q = -1.0, m = μ)
            solver = bslLD.ESChargeForm(; background = n0, variable_coefficients = vc, x_boundary = :dirichlet)
            z3() = bslLD.VectorField([bslLD.ScalarField(zeros(N, Ny)) for _ in 1:3])
            sol = bslLD.FieldSolution(z3(), z3())
            bslLD.es_quasineutral_phi!(sol, (el,), grid, solver)
            φ = sol.phi.data
            g = bslLD._es_geometry(grid; x_boundary = :dirichlet)
            nd = bslLD._species_density(el, grid, 0.0)
            nP = vc ? nd : repeat(bslLD._es_mean_x(nd, g), 1, Ny)
            r = bslLD._es_project(bslLD.moments_at(el, grid, 0.0)[1] .+ n0 .+ μ .* bslLD._es_perp(nP, φ, g), g)
            @test maximum(abs, φ[[1, N], :]) == 0
            @test maximum(abs, φ) > 1e-3
            @test maximum(abs, r) / maximum(abs, n0 .- ne) < 1e-9
        end
    end

    @testset "E×B push with walls: parity and conservation" begin
        grid, x, y, L = walls_setup(; N1 = 25, Ny = 16)
        N, Ny = length(x), length(y)
        n0 = [exp(0.1 * cos(π * x[i] / L)) * (1 + 0.2 * cos(2π * y[j] / 100)) for i in 1:N, j in 1:Ny]
        φ = [0.05 * sin(2π * x[i] / L) * cos(2π * y[j] / 100 + 0.4) + 0.03 * sin(π * x[i] / L) for i in 1:N, j in 1:Ny]
        el = bslLD.Species(bslLD.FullDK(grid, n0; T_e = 1.0, Nv = 8, exb = :rk4, dealias = true, x_boundary = :mirror);
                           q = -1.0, m = μ)
        md = getfield(el, :model)
        w = ones(N); w[[1, N]] .= 0.5                     # trapezoid weights of the wall grid
        mass(f) = sum(w .* sum(f; dims = 2:3))
        m0, f0 = mass(md.dist.data), copy(md.dist.data)
        for _ in 1:5
            bslLD.push_drift!(el, bslLD.ScalarField(φ), grid, 50.0)
        end
        ef = md.ext[2].data
        @test norm(md.dist.data .- f0) / norm(f0) > 1e-3
        @test maximum(abs, ef[N+1:end, :, :] .- ef[N-1:-1:2, :, :]) / maximum(abs, ef) < 1e-13
        @test abs(mass(md.dist.data) - m0) / m0 < 1e-13
    end

    @testset "stage equation exact with walls (variable coefficients, Euler E×B)" begin
        grid, x, y, L = walls_setup(; Ny = 8)
        N, Ny = length(x), length(y)
        dt, κ = 0.05, 1.0
        θ = 0.5 + κ * dt
        n0 = [exp(0.1 * cos(π * x[i] / L)) for i in 1:N, j in 1:Ny]
        ne = n0 .* [1 + 0.3 * cos(2π * y[j] / 100 + 0.3) * sin(2π * x[i] / L) for i in 1:N, j in 1:Ny]
        g = bslLD._es_geometry(grid; x_boundary = :dirichlet)
        el = bslLD.Species(bslLD.FullDK(grid, ne; T_e = 1.0, Nv = 4, x_boundary = :mirror); q = -1.0, m = μ)
        solver = bslLD.ESChargeForm(; background = n0, variable_coefficients = true, x_boundary = :dirichlet)
        z3() = bslLD.VectorField([bslLD.ScalarField(zeros(N, Ny)) for _ in 1:3])
        sol = bslLD.FieldSolution(z3(), z3())
        bslLD.es_quasineutral_phi!(sol, (el,), grid, solver)
        φn = copy(sol.phi.data)
        nstar = bslLD._species_density(el, grid, 0.0)
        φm = bslLD.step_es_charge_form!((el,), sol, grid, solver, dt; damping_rate = κ, corrections = 1)
        Pφ = μ .* bslLD._es_perp(nstar, (φm .- (1 - θ) .* φn) ./ θ, g)
        @test maximum(abs, φm[[1, N], :]) == 0
        @test maximum(abs, bslLD._es_project(bslLD.moments_at(el, grid, 0.0)[1] .+ n0 .+ Pφ, g)) / maximum(abs, Pφ) < 1e-8
    end
end
