@testset "Specular wall, line kernel, orbit window" begin
    maxwellian(u...) = exp(-sum(abs2, u) / 2)
    function seed(grid, fun)
        f = bslLD.Distribution(grid, 0.0)
        NX = length(grid.xaxes)
        for I in CartesianIndices(f.data)
            f.data[I] = fun(ntuple(d -> grid.xaxes[d][I[d]], NX)...,
                ntuple(d -> grid.vaxes[d][I[NX+d]], length(grid.vaxes))...)
        end
        return f
    end
    lag = bslLD.Lagrange(8)
    # node-type x axis: 16 nodes from 0 to 2π, walls on the end nodes
    grid = bslLD.Grid([0.0, -6.0, -6.0, -6.0], [2pi + 2pi / 15, 6.0, 6.0, 6.0],
        [16, 24, 24, 8], 1, 1.0, 3)
    trapz(d) = sum(d) - (sum(@view d[1, :, :, :]) + sum(@view d[end, :, :, :])) / 2

    @testset "Specular == Mirror where they must agree" begin
        # at phase 0 the specular map is the u_x flip; for f even in u_y so is Mirror
        t = bslLD.SimulationTime(0.4, 0.4)
        fun(x, a, b, c) = maxwellian(a, b, c) * (1 + 0.3a) * (1 + 0.2cos(x))
        f1, f2 = seed(grid, fun), seed(grid, fun)
        bslLD.advectX!(f1, grid, t, 1; method = lag, boundary = bslLD.Mirror())
        bslLD.advectX!(f2, grid, t, 1; method = lag, boundary = bslLD.Specular())
        @test f1.data == f2.data

        # along B both flip v_z only
        g3 = bslLD.Grid([0.0, 0.0, 0.0, -3.0, -3.0, -3.0], [1.0, 1.0, 2pi + 2pi / 11, 3.0, 3.0, 3.0],
            [1, 1, 12, 6, 6, 6], 3, 1.0, 3)
        t.phase = 0.9
        h(x, y, z, a, b, c) = maxwellian(a, b, c) * (1 + 0.3c + 0.1a) * (1 + 0.2cos(z))
        f1, f2 = seed(g3, h), seed(g3, h)
        bslLD.advectX!(f1, g3, t, 3; method = lag, boundary = bslLD.Mirror())
        bslLD.advectX!(f2, g3, t, 3; method = lag, boundary = bslLD.Specular())
        @test f1.data == f2.data
    end

    @testset "Specular at arbitrary phase" begin
        t = bslLD.SimulationTime(0.4, 0.4)
        t.phase = 0.7
        # an isotropic, uniform state is invariant (up to the velocity interpolation
        # of the rotated halo)
        f = seed(grid, (x, a, b, c) -> maxwellian(a, b, c))
        before = copy(f.data)
        bslLD.advectX!(f, grid, t, 1; method = lag, boundary = bslLD.Specular(),
            max_wall_shift = Inf)                       # one sweep, one remap
        @test maximum(abs, f.data .- before) < 5e-4
        # mass in the reflecting-domain norm, anisotropic f, rotating phase
        f = seed(grid, (x, a, b, c) -> maxwellian(a - 0.5, b, c) * (1 + 0.3sin(x)))
        m0 = trapz(f.data)
        for k = 1:50
            t.phase = 0.37k
            bslLD.advectX!(f, grid, t, 1; method = lag, boundary = bslLD.Specular())
        end
        @test abs(trapz(f.data) / m0 - 1) < 1e-4
        # the velocity remap is g(u) = f(R(2φ) F u): a Maxwellian centred at (1, 0)
        # maps to one centred at (-cos 2φ, -sin 2φ)
        f = seed(grid, (x, a, b, c) -> maxwellian(a - 1, b, c))
        plan = bslLD._get_plan(f.dist, grid)
        for phase in (0.3, 1.2, 2.0)
            t.phase = phase
            ctx = bslLD._x_context(f, grid, t, 1, plan)
            r, fd = bslLD._specular_remap(copy(f.data), ctx, lag, bslLD.backend())
            g = reverse(r; dims = 1 + fd)[1, :, :, :]
            m1 = sum(g .* reshape(collect(grid.vaxes[1]), :, 1, 1)) / sum(g)
            m2 = sum(g .* reshape(collect(grid.vaxes[2]), 1, :, 1)) / sum(g)
            @test m1 ≈ -cos(2phase) atol = 1e-3
            @test m2 ≈ -sin(2phase) atol = 1e-3
        end
        @test_throws ArgumentError bslLD.advectX!(
            f, grid, t, 1; method = bslLD.Fourier(), boundary = bslLD.Specular())
    end

    @testset "line kernel == per-point kernel" begin
        g2 = bslLD.Grid([0.0, 0.0, -4.0, -4.0, -4.0], [2pi + 2pi / 11, 2pi, 4.0, 4.0, 4.0],
            [12, 6, 8, 8, 10], 2, 1.0, 3)
        f = bslLD.Distribution(g2, 0.0)
        f.data .= rand(size(f.data)...)
        t = bslLD.SimulationTime(0.7, 0.7)
        t.phase = 0.4
        plan = bslLD._get_plan(f.dist, g2)
        exec = bslLD.backend()
        E = bslLD.VectorField([rand(12, 6) for _ = 1:3])
        for (ctx, bc) in ((bslLD._x_context(f, g2, t, 1, plan), bslLD.Mirror()),
                          (bslLD._x_context(f, g2, t, 2, plan), bslLD.Periodic()),
                          (bslLD._v_context(f, g2, t, E, 3, plan), bslLD.Periodic()))
            a = similar(f.data)
            bslLD.lagrange_shift_kernel!(exec)(a, f.data, ctx, Val(8), bc; ndrange = length(a))
            bslLD.KernelAbstractions.synchronize(exec)
            b = similar(f.data)
            bslLD._lagrange_sweep!(b, f.data, ctx, lag, bc, exec)
            @test a == b
        end
    end

    @testset "wall sub-cycling" begin
        # shift of 3.2 cells: four sub-sweeps of <= 1 cell, identical to four explicit
        # quarter steps; periodic axes are never sub-cycled
        g1 = bslLD.Grid([0.0, -4.0, -4.0, -1.0], [2pi + 2pi / 19, 4.0, 4.0, 1.0], [20, 8, 8, 2], 1, 1.0, 3)
        t = bslLD.SimulationTime(0.25, 0.25)
        f = seed(g1, (x, a, b, c) -> maxwellian(a, b, c) * (1 + 0.3a + 0.2cos(x)))
        ctx = bslLD._x_context(f, g1, t, 1, bslLD._get_plan(f.dist, g1))
        @test bslLD._wall_subcycles(bslLD.Mirror(), ctx, 1.0) == ceil(Int, 0.25 * 4 / (2pi / 19))
        @test bslLD._wall_subcycles(bslLD.Periodic(), ctx, 1.0) == 1
        n = bslLD._wall_subcycles(bslLD.Mirror(), ctx, 1.0)
        f2 = deepcopy(f)
        bslLD.advectX!(f, g1, t, 1; method = lag, boundary = bslLD.Mirror())
        t.fraction_dt = 1 / n
        for _ = 1:n
            bslLD.advectX!(f2, g1, t, 1; method = lag, boundary = bslLD.Mirror())
        end
        @test f.data == f2.data
    end

    @testset "orbit window" begin
        # x-shift of cos(k x) over a window w centred on the phase: the gyration-plane
        # displacement is dt sinc(Ω w/2) (Q u)_x, the field-aligned one unchanged
        g1 = bslLD.Grid([0.0, -1.5, -1.5, -1.5], [2pi, 1.5, 1.5, 1.5], [16, 4, 4, 4], 1, 1.0, 3)
        t = bslLD.SimulationTime(0.8, 0.8)
        t.phase = 0.6
        f = seed(g1, (x, a, b, c) -> cos(x))
        bslLD.advectX!(f, g1, t, 1; orbit_window = 0.8)
        fac = sin(0.4) / 0.4
        expected = seed(g1, (x, a, b, c) -> cos(x - 0.8 * fac * (cos(0.6) * a + sin(0.6) * b)))
        @test maximum(abs, f.data .- expected.data) < 1e-12
        @test bslLD._orbit_factor(bslLD.Distribution(g1, 0.0), g1, 0) == 1
    end
end
