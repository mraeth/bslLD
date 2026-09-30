@testset "Magnetic geometry" begin
    maxwellian(u...) = exp(-sum(abs2, u) / 2) / (2pi)^(length(u) / 2)

    function grid2x3v(; nx = 4, ny = 8, nv = 4, vmax = 1.5, ly = 2pi)
        return bslLD.Grid([-2.0, 0.0, -vmax, -vmax, -vmax], [2.0, ly, vmax, vmax, vmax],
            [nx, ny, nv, nv, nv], 2, 1.0, 3)
    end

    function seed(grid, fun)
        f = bslLD.Distribution(grid, 0.0)
        for I in CartesianIndices(f.data)
            NX = length(grid.xaxes)
            xs = ntuple(d -> grid.xaxes[d][I[d]], NX)
            vs = ntuple(d -> grid.vaxes[d][I[NX+d]], 3)
            f.data[I] = fun(xs, vs)
        end
        return f
    end

    @testset "ShearedSlab y-characteristic" begin
        grid = grid2x3v()
        ky, Ls, R0 = 1.0, 3.0, 7.0
        t = bslLD.SimulationTime(0.4, 0.4)
        g = bslLD.ShearedSlab(; Ls, R0)
        init((x, y), v) = cos(ky * y) * (1 + 0.1v[1])
        expected = seed(grid, ((x, y), (u1, u2, u3)) -> begin
            vy = u2 + x / Ls * u3 - (u3^2 + (u1^2 + u2^2) / 2) / R0
            init((x, y - t.dt * vy), (u1, u2, u3))
        end).data

        f = seed(grid, init)
        bslLD.advectX!(f, grid, t, 2; geometry = g)
        @test maximum(abs, f.data .- expected) < 1e-12

        f = seed(grid, init)
        bslLD.advectX!(f, grid, t, 2; method = bslLD.Lagrange(8), geometry = g)
        @test maximum(abs, f.data .- expected) < 1e-3

        # x-advection and the unsheared slab are untouched
        f1, f2 = seed(grid, init), seed(grid, init)
        bslLD.advectX!(f1, grid, t, 1; geometry = g)
        bslLD.advectX!(f2, grid, t, 1)
        @test f1.data == f2.data
        f3 = seed(grid, init)
        bslLD.advectX!(f3, grid, t, 2; geometry = bslLD.ShearedSlab())
        bslLD.advectX!(f2, grid, t, 2)
        @test maximum(abs, f3.data .- f2.data) < 1e-14

        bad = bslLD.Grid([0.0, 0.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0], [4, 4, 4, 4], 2)
        @test_throws ArgumentError bslLD.advectX!(
            bslLD.Distribution(bad, 0.0), bad, t, 2; geometry = g)
    end

    @testset "CurvedPatch toroidal metric" begin
        grid = bslLD.Grid([-2.0, 0.0, 0.0, -1.5, -1.5, -1.5], [2.0, 1.0, 2pi, 1.5, 1.5, 1.5],
            [4, 2, 8, 4, 4, 4], 3, 1.0, 3)
        Rc = 5.0
        t = bslLD.SimulationTime(0.3, 0.3)
        init((x, y, z), v) = sin(z) * (1 + 0.1v[3])
        expected = seed(grid, ((x, y, z), v) -> init((x, y, z - t.dt * v[3] * Rc / (Rc + x)), v))
        f = seed(grid, init)
        bslLD.advectX!(f, grid, t, 3; geometry = bslLD.CurvedPatch(; Rc))
        @test maximum(abs, f.data .- expected.data) < 1e-12
    end

    @testset "CurvedPatch velocity-space forces" begin
        grid = grid2x3v(; nx = 5, ny = 1, nv = 40, vmax = 6.0)
        x = grid.xaxes[1]
        dv3 = prod(grid.delta[3:5])
        mean_u(f, d) = [sum(f.data[i, :, :, :, :] .* reshape(collect(grid.vaxes[d]),
            ntuple(k -> k == d + 1 ? length(grid.vaxes[d]) : 1, 4))) /
                        sum(f.data[i, :, :, :, :]) for i in eachindex(x)]

        t = bslLD.SimulationTime(0.5, 0.5)
        Rc = 5.0
        g = bslLD.CurvedPatch(; Rc)

        # no-op for the slab geometries
        f = seed(grid, (xs, v) -> maxwellian(v...))
        before = copy(f.data)
        bslLD.advect_geometry!(f, grid, t, bslLD.Slab())
        bslLD.advect_geometry!(f, grid, t, bslLD.ShearedSlab(; R0 = 3.0))
        @test f.data == before

        # an isotropic Maxwellian is an equilibrium of both forces; what is left
        # is the O(h^3 / Rc^2) splitting error of kick-scaling-kick
        bslLD.advect_geometry!(f, grid, t, g)
        @test maximum(abs, f.data .- before) < 5e-4

        # residual field: rotation of (u_x, u_y) by θ = (Rc/(Rc+x) - 1) h,
        # u = (1, 0) -> u_y = -sin θ; the curvature only feeds u_x at phase 0
        f = seed(grid, (xs, v) -> maxwellian(v[1] - 1, v[2], v[3]))
        bslLD.advect_geometry!(f, grid, t, g)
        theta = @. (Rc / (Rc + x) - 1) * t.dt
        @test maximum(abs, mean_u(f, 2) .+ sin.(theta)) < 1e-5

        # curvature, advective form: f is constant along characteristics, so at
        # fixed x the velocity flow is compressible (∂_u·a = -v_x/(Rc+x)) and
        # d<u_x>/dt = (<u_z^2> - <u_x^2>)/(Rc+x). Mass balances only together with
        # the x-advection, in the measure J = (Rc+x)/Rc.
        f = seed(grid, (xs, v) -> maxwellian(v[1], v[2], v[3] - 1))   # <u_z^2> = 2
        bslLD.advect_geometry!(f, grid, t, g)
        @test maximum(abs, mean_u(f, 1) ./ (t.dt ./ (Rc .+ x)) .- 1) < 0.05
    end

    @testset "Adiabatic field with mirror x" begin
        n, L = 17, 3.0
        grid = bslLD.Grid([0.0, 0.0], [L + L / (n - 1), 2pi], [n, 12], 2, 1.0, 3)
        x, y = grid.xaxes
        @test x[end] ≈ L
        rho = bslLD.ScalarField([cos(pi * xi / L) + cos(2yi) for xi in x, yi in y])
        sol = bslLD.solve_fields(bslLD.Moments(rho), grid,
            bslLD.AdiabaticSolver((bslLD.Mirror(), bslLD.Periodic())))
        @test maximum(abs, sol.E[1].data .- [pi / L * sin(pi * xi / L) for xi in x, yi in y]) <
              1e-12
        @test maximum(abs, sol.E[2].data .- [2sin(2yi) for xi in x, yi in y]) < 1e-12
        # the default is unchanged
        @test bslLD.AdiabaticSolver() == bslLD.AdiabaticSolver(())
    end

    @testset "kappa_n drive" begin
        grid = grid2x3v(; nx = 2, ny = 2, nv = 6)
        f = seed(grid, (xs, v) -> 0.0)
        Ey = 0.3
        E = bslLD.VectorField([fill(0.0, 2, 2), fill(Ey, 2, 2), fill(0.0, 2, 2)])
        bslLD.add_kappaT!(f, grid, 0.1, 0.0, E; kappa_n = 0.5)
        expected = seed(grid, (xs, v) -> 0.1 * 0.5 * Ey * maxwellian(v...))
        @test maximum(abs, f.data .- expected.data) < 1e-14
    end
end
