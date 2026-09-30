@testset "Line kernel, wall sub-cycling, orbit window" begin
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
