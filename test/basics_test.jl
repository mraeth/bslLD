using Adapt

@testset "Basics" begin
    bslLD.set_execution_space!(exec = bslLD.backend())

    @testset "Grid" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 1)
        simTime = bslLD.SimulationTime(0.05, 5.0; nmax = 100)

        @test simTime.dt == 0.05
        @test simTime.final_T == 5.0
        @test simTime.nmax == 100
        @test length(grid.xaxes) == 1
        @test length(grid.vaxes) == 1
        @test length(grid.xaxes[1]) == 4
        @test length(grid.vaxes[1]) == 9
        @test grid.b0 == 1.0
        @test grid.max[1] == 10.0
        @test grid.min[2] == -6.0
        @test grid.delta[1] == 2.5
        @test grid.xaxes isa Tuple
        @test grid.vaxes isa Tuple

        bslLD.advance!(simTime, 0.0)
        @test simTime.step == 1
        @test simTime.current_T == 0.05
    end

    @testset "Grid Adaptation" begin
        grid = bslLD.Grid([0.0, -2.0], [2pi, 2.0], [8, 4], 1)
        adapted = Adapt.adapt(Array, grid)

        @test adapted.delta == grid.delta
        @test adapted.xaxes == grid.xaxes
        @test adapted.vaxes == grid.vaxes
        @test adapted.b0 == grid.b0
        @test adapted.Bdir == grid.Bdir
    end

    @testset "Distribution" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 1)
        f = bslLD.Distribution(grid, 0.01)

        @test size(f.data) == (length(grid.xaxes[1]), length(grid.vaxes[1]))
        @test f.data isa Array
        @test f.m == 1.0
        @test f.q == 1.0

        ion = bslLD.Distribution(grid, 0.01; m = 4.0, q = 2.0)
        @test ion.m == 4.0
        @test ion.q == 2.0
        @test bslLD.thermal_velocity(ion) == 0.5
        @test bslLD.electric_acceleration_scale(ion) == 1.0
        @test_throws ArgumentError bslLD.Distribution(grid, 0.01; m = 0.0)
    end

    @testset "Fields" begin
        grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [4, 8], 1)
        f = bslLD.Distribution(grid, 0.01)
        rho = bslLD.compute_density(f, grid)

        @test size(rho.data) == (length(grid.xaxes[1]),)

        scalar = bslLD.ScalarField(reshape(collect(1.0:4.0), 2, 2))
        vector = bslLD.VectorField([collect(1.0:4.0), collect(5.0:8.0)])

        @test scalar.data isa Array
        @test all(component -> component isa bslLD.ScalarField, vector)
        @test all(component -> component.data isa Array, vector)
    end
end

@testset "deposit! and gaussian_ramp" begin
    grid = bslLD.Grid([0.0, -4.0], [2pi, 4.0], [4, 8], 1)
    zero_f() = bslLD.Distribution(grid, 0.0; initFuncx = x -> 0.0)
    target(grid, t) = fill!(similar(zero_f().data), 1.0 + t)

    f = zero_f()
    @test all(iszero, Array(f.data))

    # initial condition: weight 1, dt 1
    @test bslLD.deposit!(f, grid, target, 0.0; weight = 1, dt = 1) === f
    @test all(==(1.0), Array(f.data))

    # loop use: weight * dt * target(grid, t)
    bslLD.deposit!(f, grid, target, 2.0; weight = 0.5, dt = 0.2)
    @test all(≈(1.0 + 0.5 * 0.2 * 3.0), Array(f.data))

    # zero weight: no change and target is not evaluated
    g = zero_f()
    bslLD.deposit!(
        g,
        grid,
        (grid, t) -> error("must not be called"),
        1.0;
        weight = 0.0,
        dt = 0.1,
    )
    @test all(iszero, Array(g.data))

    # wrong target size
    @test_throws DimensionMismatch bslLD.deposit!(
        g,
        grid,
        (grid, t) -> ones(2, 2),
        0.0;
        weight = 1,
        dt = 1,
    )

    # a Species and the bare distribution behave the same
    h = zero_f()
    bslLD.deposit!(h.dist, grid, target, 0.0; weight = 1, dt = 1)
    @test Array(h.data) == Array(zero_f().data .+ 1.0)

    # ramp: integral over the window is 1, zero outside, peak at t_mid
    w = bslLD.gaussian_ramp(15.0, 3.0)
    dts = 0.01
    ts = (0.5dts):dts:(30.0-0.5dts)
    @test isapprox(sum(w, ts) * dts, 1.0; atol = 1e-6)
    @test w(-0.1) == 0 && w(30.0) == 0 && w(30.1) == 0
    @test w(15.0) > w(12.0) > w(5.0)
    @test_throws ArgumentError bslLD.gaussian_ramp(-1.0, 3.0)

    # deposit! in a ramp loop builds f up to amplitude * (sum of weights * dt) = 1 * amplitude
    f = zero_f()
    dt = 0.05
    t = 0.0
    while t < 30.0
        bslLD.deposit!(
            f,
            grid,
            (grid, t) -> fill!(similar(f.data), 2.0),
            t;
            weight = w(t + dt / 2),
            dt,
        )
        t += dt
    end
    @test all(x -> isapprox(x, 2.0; atol = 1e-4), Array(f.data))
end
