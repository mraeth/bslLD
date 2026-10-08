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

    @testset "CurvedPatch pitch metric" begin
        grid = bslLD.Grid([-2.0, 0.0, 0.0, -1.5, -1.5, -1.5], [2.0, 2pi, 2pi, 1.5, 1.5, 1.5],
            [4, 8, 8, 4, 4, 4], 3, 1.0, 3)
        g = bslLD.CurvedPatch(; Rc = 8.0, r0 = 5.0, q0 = 1.5, shat = 0.6)
        @test g.R0 == 3.0
        t = bslLD.SimulationTime(0.3, 0.3)
        t.phase = 0.7          # v_2 = -sin(φ) u_x + cos(φ) u_y
        s, c = sincos(t.phase)
        init((x, y, z), v) = sin(y + z) * (1 + 0.1v[3])
        for dir in (2, 3)
            expected = seed(grid, function ((x, y, z), v)
                m22, m23, m33 = bslLD._patch_metric(g, x)
                v2 = c * v[2] - s * v[1]
                d = dir == 2 ? m22 * v2 + m23 * v[3] : m23 * v2 + m33 * v[3]
                return init((x, dir == 2 ? y - t.dt * d : y, dir == 3 ? z - t.dt * d : z), v)
            end)
            f = seed(grid, init)
            bslLD.advectX!(f, grid, t, dir; geometry = g)
            @test maximum(abs, f.data .- expected.data) < 1e-12
        end
        # the metric matrix maps the logical gradient to the physical field
        E = bslLD.VectorField([fill(0.0, 4, 8, 8), fill(1.0, 4, 8, 8), fill(0.5, 4, 8, 8)])
        bslLD.apply_metric!(E, grid, g)
        m = bslLD._patch_metric.(Ref(g), grid.xaxes[1])
        @test E[2].data[:, 1, 1] ≈ getindex.(m, 1) .+ 0.5 .* getindex.(m, 2)
        @test E[3].data[:, 1, 1] ≈ getindex.(m, 2) .+ 0.5 .* getindex.(m, 3)
        # 2D grid: the logical E_3 is zero, whatever the reused buffer holds, so
        # repeated solves + apply_metric! do not accumulate E_3
        g2 = grid2x3v(; nx = 5, ny = 8)
        rho = bslLD.ScalarField([0.1 * sin(y) for x in g2.xaxes[1], y in g2.xaxes[2]])
        Es = map(1:3) do _
            E = bslLD.solve_fields(bslLD.Moments(rho), g2, bslLD.AdiabaticSolver()).E
            copy(bslLD.apply_metric!(E, g2, g)[3].data)
        end
        m2 = bslLD._patch_metric.(Ref(g), g2.xaxes[1])
        @test Es[3] == Es[1]
        @test Es[1] ≈ getindex.(m2, 2) .* [-0.1 * cos(y) for x in g2.xaxes[1], y in g2.xaxes[2]]
        # r0 + x must stay positive; q0 needs a finite r0
        @test_throws ArgumentError bslLD.advectX!(seed(grid, init), grid, t, 2;
            geometry = bslLD.CurvedPatch(; Rc = 8.0, r0 = 1.0))
        @test_throws ArgumentError bslLD.CurvedPatch(; Rc = 8.0, q0 = 2.0)
    end

    @testset "CurvedPatch velocity-space forces" begin
        grid = grid2x3v(; nx = 5, ny = 1, nv = 40, vmax = 6.0)
        x = grid.xaxes[1]
        mean_u(f, d) = [sum(f.data[i, :, :, :, :] .* reshape(collect(grid.vaxes[d]),
            ntuple(k -> k == d + 1 ? length(grid.vaxes[d]) : 1, 4))) /
                        sum(f.data[i, :, :, :, :]) for i in eachindex(x)]
        lag = bslLD.Lagrange(8)

        t = bslLD.SimulationTime(0.5, 0.5)
        Rc = 5.0
        g = bslLD.CurvedPatch(; Rc)
        gp = bslLD.CurvedPatch(; Rc = 15.0, r0 = 10.0, q0 = 1.5, shat = 0.8)

        # no-op for the slab geometries
        f = seed(grid, (xs, v) -> maxwellian(v...))
        before = copy(f.data)
        bslLD.advect_geometry!(f, grid, t, bslLD.Slab())
        bslLD.advect_geometry!(f, grid, t, bslLD.ShearedSlab(; R0 = 3.0))
        @test f.data == before

        # an isotropic Maxwellian is an equilibrium of both forces; what is left
        # is the O(h^3 / R^2) splitting error of the curvature sweeps
        for geo in (g, gp)
            f = seed(grid, (xs, v) -> maxwellian(v...))
            bslLD.advect_geometry!(f, grid, t, geo)
            @test maximum(abs, f.data .- before) < 5e-4
        end

        # residual field, no pitch: rotation of (u_x, u_y) by θ = (Rc/(Rc+x) - 1) h,
        # u = (1, 0) -> u_y = -sin θ
        f = seed(grid, (xs, v) -> maxwellian(v[1] - 1, v[2], v[3]))
        bslLD._residual_rotation!(f, grid, g, t.dt, 1.0, 0.0, 1.0, lag)
        theta = @. (Rc / (Rc + x) - 1) * t.dt
        @test maximum(abs, mean_u(f, 2) .+ sin.(theta)) < 1e-5

        # residual field with pitch: an exact rotation by ρ = -h (β2 ê + (β3 - 1) ẑ),
        # ê = (-sin φ, cos φ, 0); compare the mean velocity with Rodrigues' formula
        U = [1.0, 0.5, -0.5]
        phi = 0.4
        f = seed(grid, (xs, v) -> maxwellian(v[1] - U[1], v[2] - U[2], v[3] - U[3]))
        bslLD._residual_rotation!(f, grid, gp, t.dt, cos(phi), sin(phi), 1.0, lag)
        for (i, xi) in enumerate(x)
            b2, b3 = bslLD._patch_field(gp, xi)
            rho = -t.dt .* [-sin(phi) * b2, cos(phi) * b2, b3 - 1]
            th = sqrt(sum(abs2, rho))
            k = rho ./ th
            Urot = U .* cos(th) .+ [k[2] * U[3] - k[3] * U[2], k[3] * U[1] - k[1] * U[3], k[1] * U[2] - k[2] * U[1]] .* sin(th) .+ k .* sum(k .* U) .* (1 - cos(th))
            @test maximum(abs, [mean_u(f, d)[i] for d = 1:3] .- Urot) < 1e-5
        end

        # curvature, advective form: f is constant along characteristics, so at
        # fixed x the velocity flow is compressible and, at phase 0,
        # d<u_x>/dt = (<v_θ²> - <u_x²>)/(r0+x) + (<v_φ²> - <u_x²>)/(Rc+x).
        # Mass balances only together with the x-advection, in the measure J.
        f = seed(grid, (xs, v) -> maxwellian(v[1], v[2], v[3] - 1))   # <u_z^2> = 2
        bslLD.advect_geometry!(f, grid, t, g)
        @test maximum(abs, mean_u(f, 1) ./ (t.dt ./ (Rc .+ x)) .- 1) < 0.05
        # poloidal pair alone (no pitch, v_θ = u_y) is the toroidal pair with
        # u_y and u_z swapped (L = 20: at small L the departure points of the
        # scaling leave the periodic velocity box and wrap, differently per sweep)
        f = seed(grid, (xs, v) -> maxwellian(v[1], v[2], v[3] - 1))
        bslLD._curvature_force!(f, grid, bslLD.CurvedPatch(; Rc = 20.0), t.dt, 1.0, 0.0, lag)
        tor = (mean_u(f, 1), mean_u(f, 3))
        f = seed(grid, (xs, v) -> maxwellian(v[1], v[2] - 1, v[3]))
        bslLD._curvature_force!(f, grid, bslLD.CurvedPatch(; Rc = 1e12, r0 = 20.0), t.dt, 1.0,
            0.0, lag)
        @test maximum(abs, mean_u(f, 1) .- tor[1]) < 1e-4
        @test maximum(abs, mean_u(f, 2) .- tor[2]) < 1e-4
        # with pitch, u_z has a poloidal part: <v_θ²> = 1 + sin²α, <v_φ²> = 1 + cos²α
        sa, ca = sincos(atan(bslLD._pitch(gp)))
        f = seed(grid, (xs, v) -> maxwellian(v[1], v[2], v[3] - 1))
        bslLD._curvature_force!(f, grid, gp, t.dt, 1.0, 0.0, lag)
        expected = @. t.dt * (sa^2 / (gp.r0 + x) + ca^2 / (gp.Rc + x))
        @test maximum(abs, mean_u(f, 1) ./ expected .- 1) < 0.05
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
