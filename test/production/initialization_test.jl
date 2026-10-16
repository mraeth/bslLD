using bslLD.Production

@testset "deposit! and gaussian_ramp" begin
    grid = bslLD.Grid([0.0, -4.0], [2pi, 4.0], [4, 8], 1)
    zero_f() = bslLD.Distribution(grid, 0.0; initFuncx = x -> 0.0)
    target(grid, t) = fill!(similar(zero_f().data), 1.0 + t)

    f = zero_f()
    @test all(iszero, Array(f.data))

    # initial condition: weight 1, dt 1
    @test Production.deposit!(f, grid, target, 0.0; weight = 1, dt = 1) === f
    @test all(==(1.0), Array(f.data))

    # loop use: weight * dt * target(grid, t)
    Production.deposit!(f, grid, target, 2.0; weight = 0.5, dt = 0.2)
    @test all(≈(1.0 + 0.5 * 0.2 * 3.0), Array(f.data))

    # zero weight: no change and target is not evaluated
    g = zero_f()
    Production.deposit!(
        g,
        grid,
        (grid, t) -> error("must not be called"),
        1.0;
        weight = 0.0,
        dt = 0.1,
    )
    @test all(iszero, Array(g.data))

    # wrong target size
    @test_throws DimensionMismatch Production.deposit!(
        g,
        grid,
        (grid, t) -> ones(2, 2),
        0.0;
        weight = 1,
        dt = 1,
    )

    # a Species and the bare distribution behave the same
    h = zero_f()
    Production.deposit!(h.dist, grid, target, 0.0; weight = 1, dt = 1)
    @test Array(h.data) == Array(zero_f().data .+ 1.0)

    # ramp: integral over the window is 1, zero outside, peak at t_mid
    w = Production.gaussian_ramp(15.0, 3.0)
    dts = 0.01
    ts = (0.5dts):dts:(30.0-0.5dts)
    @test isapprox(sum(w, ts) * dts, 1.0; atol = 1e-6)
    @test w(-0.1) == 0 && w(30.0) == 0 && w(30.1) == 0
    @test w(15.0) > w(12.0) > w(5.0)
    @test_throws ArgumentError Production.gaussian_ramp(-1.0, 3.0)

    # deposit! in a ramp loop builds f up to amplitude * (sum of weights * dt) = 1 * amplitude
    f = zero_f()
    dt = 0.05
    t = 0.0
    while t < 30.0
        Production.deposit!(
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
