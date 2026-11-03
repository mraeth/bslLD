# Polar perpendicular-velocity grid (PolarVGrid, P2 scheme): quadrature, init, moments,
# x advection, E-kicks K1 (PolarSL) and K2 (PolarTaylor).

_pgauss(vx, vy, c, s) = exp(-((vx - c[1])^2 + (vy - c[2])^2) / (2s^2))
_pvx(g, iv) = g.vr[iv] * cos(g.vphi[iv])
_pvy(g, iv) = g.vr[iv] * sin(g.vphi[iv])

function _ptranslate_err(Ns, Nphi, method; c, d, s = 0.6)
    g = bslLD.PolarVGrid(0.0, 1.0, 2, Ns, Nphi; rmax = 6.0)
    f = [_pgauss(_pvx(g, iv), _pvy(g, iv), c, s) for ix = 1:2, iv = 1:g.Nv]
    m0 = sum(f[1, :] .* g.weights)
    bslLD.polar_kick!(f, g, d[1], d[2]; method = method)
    ex = [_pgauss(_pvx(g, iv) - d[1], _pvy(g, iv) - d[2], c, s) for iv = 1:g.Nv]
    return maximum(abs, f[1, :] .- ex), abs(sum(f[1, :] .* g.weights) / m0 - 1)
end

@testset "PolarVGrid" begin
    @testset "endpoint corrections and quadrature" begin
        @test bslLD.polar_endpoint_corrections(1) == [-1 // 12]
        # plain midpoint is O(h^2) for the odd integrand; K = 6 is O(h^14)
        err(K, Ns, p) = begin
            g = bslLD.PolarVGrid(0.0, 1.0, 1, Ns, 4; rmax = 8.0, map = bslLD.IdentityMap(), quad_order = K)
            sp = bslLD.Distribution(g, 0.0; initFuncv = (r, φ) -> r^p * exp(-r^2 / 2) / 2pi)
            bslLD.compute_density(sp, g).data[1] - (p == 0 ? 1.0 : 2.0)
        end
        @test 3.8 < err(0, 32, 0) / err(0, 64, 0) < 4.2
        @test abs(err(6, 32, 0)) < 1e-9
        @test abs(err(6, 64, 0)) < 1e-13
        @test abs(err(6, 64, 2)) < 1e-11
        # sinh map: same accuracy with fewer radial nodes
        g = bslLD.PolarVGrid(0.0, 1.0, 1, 24, 4; rmax = 8.0, map = bslLD.SinhMap(2.0))
        @test abs(bslLD.compute_density(bslLD.Distribution(g, 0.0), g).data[1] - 1) < 1e-8
    end

    @testset "init and moments" begin
        L = 2pi / 1.5
        g = bslLD.PolarVGrid(0.0, L, 8, 32, 16; rmax = 8.0)
        sp = bslLD.Distribution(g, 1e-3)
        x = g.xaxes[1]
        @test maximum(abs, bslLD.compute_density(sp, g).data .- (1 .+ 1e-3 .* cos.(1.5 .* x))) < 1e-10
        # current of a shifted Maxwellian (rotating frame), and lab-frame rotation
        u = (0.3, -0.2)
        sp2 = bslLD.Distribution(g, 0.0; initFuncv = (r, φ) -> _pgauss(r * cos(φ), r * sin(φ), u, 1.0) / 2pi)
        J = bslLD.compute_current(sp2, g)
        @test maximum(abs, J[1].data .- u[1]) < 1e-9
        @test maximum(abs, J[2].data .- u[2]) < 1e-9
        th = 0.4
        Jl = bslLD.compute_current(sp2, g, th)
        @test maximum(abs, Jl[1].data .- (cos(-th) * u[1] - sin(-th) * u[2])) < 1e-9
    end

    @testset "x advection: analytic and Cartesian convention" begin
        k = 1.5
        L = 2pi / k
        g = bslLD.PolarVGrid(0.0, L, 16, 12, 16; rmax = 5.0)
        x = g.xaxes[1]
        eps = 0.1
        sp = bslLD.Distribution(g, eps)
        st = bslLD.SimulationTime(0.3, 1.0; phase = 0.7)
        st.fraction_dt = 0.5
        bslLD.advectX!(sp, g, st)
        tau, th = 0.15, 0.7
        ex = [
            (1 + eps * cos(k * (x[ix] - tau * g.vr[iv] * cos(g.vphi[iv] - th)))) *
            exp(-g.vr[iv]^2 / 2) / 2pi for ix in eachindex(x), iv = 1:g.Nv
        ]
        @test maximum(abs, sp.data .- ex) < 1e-14
        # the Cartesian path follows the same formula with r cos(φ-θ) = vx cosθ + vy sinθ
        cg = bslLD.Grid([0.0, -3.0, -3.0], [L, 3.0, 3.0], [16, 6, 6], 1)
        cf = bslLD.Distribution(cg, eps; initFuncx = x -> 1 + eps * cos(k * x))
        bslLD.advectX!(cf, cg, st)
        cx, vx, vy = cg.xaxes[1], cg.vaxes[1], cg.vaxes[2]
        cex = [
            (1 + eps * cos(k * (cx[i] - tau * (vx[a] * cos(th) + vy[b] * sin(th))))) *
            exp(-(vx[a]^2 + vy[b]^2) / 2) / 2pi for i in eachindex(cx), a in eachindex(vx),
            b in eachindex(vy)
        ]
        @test maximum(abs, cf.data .- cex) < 1e-13
    end

    @testset "E kick: Cartesian convention" begin
        th, tau = 0.9, 0.25
        E = [0.8, -0.4]
        dl = tau .* (cos(th) * E[1] - sin(th) * E[2], sin(th) * E[1] + cos(th) * E[2])
        e = bslLD.VectorField([fill(E[1], 2), fill(E[2], 2), zeros(2)])
        st = bslLD.SimulationTime(tau, 1.0; phase = th)
        c0 = (0.5, -0.3)
        # Cartesian: f(v) <- f(v - δ) with δ = τ R(θ) E (q = m = 1)
        cg = bslLD.Grid([0.0, -6.0, -6.0], [1.0, 6.0, 6.0], [2, 64, 64], 1)
        cf = bslLD.Distribution(cg, 0.0; initFuncx = x -> 1.0, initFuncv = v -> 1.0)
        vx, vy = cg.vaxes
        cf.data .= [_pgauss(vx[a], vy[b], c0, 0.6) for i = 1:2, a in eachindex(vx), b in eachindex(vy)]
        bslLD.advectV!(cf, cg, st, e)
        cex = [_pgauss(vx[a] - dl[1], vy[b] - dl[2], c0, 0.6) for i = 1:2, a in eachindex(vx), b in eachindex(vy)]
        @test maximum(abs, cf.data .- cex) < 1e-10
        g = bslLD.PolarVGrid(0.0, 1.0, 2, 64, 64; rmax = 6.0)
        ex = [_pgauss(_pvx(g, iv) - dl[1], _pvy(g, iv) - dl[2], c0, 0.6) for ix = 1:2, iv = 1:g.Nv]
        for method in (bslLD.PolarSL(8), bslLD.PolarTaylor(8))
            sp = bslLD.Distribution(g, 0.0; initFuncx = x -> 1.0, initFuncv = (r, φ) -> _pgauss(r * cos(φ), r * sin(φ), c0, 0.6))
            bslLD.advectV!(sp, g, st, e; method = method)
            @test maximum(abs, sp.data .- ex) < 1e-7
        end
    end

    @testset "no-op and rotation equivariance" begin
        g = bslLD.PolarVGrid(0.0, 1.0, 3, 16, 24; rmax = 6.0)
        f0 = [_pgauss(_pvx(g, iv), _pvy(g, iv), (0.4, 0.2), 0.7) * (1 + 0.1ix) for ix = 1:3, iv = 1:g.Nv]
        for method in (bslLD.PolarSL(6), bslLD.PolarTaylor(6))
            f = copy(f0)
            bslLD.polar_kick!(f, g, 0.0, 0.0; method = method)
            @test f == f0
        end
        # K1 evaluated at δ = 0 reproduces the nodes (Fourier sum + Lagrange delta stencil)
        ws = bslLD._k1_workspaces(g, Float64, Val(6), 1)[1]
        for l = 1:24, j = 1:16
            ws.g[j, l] = f0[1, g.offset[j]+l]
        end
        @test maximum(abs, bslLD._k1_slice!(ws, g, 0.0, 0.0, Val(6)) .- reshape(f0[1, :], 24, 16)') < 1e-14
        # kick(rot_n f, δ) == rot_n kick(f, R(-α) δ),  α = n Δφ
        n, dl = 5, (0.3, -0.15)
        α = 2pi * n / 24
        rot(f) = (fr = similar(f); for j = 1:16, l = 1:24
            fr[:, g.offset[j]+l] = f[:, g.offset[j]+mod1(l - n, 24)]
        end; fr)
        dlr = (cos(α) * dl[1] + sin(α) * dl[2], -sin(α) * dl[1] + cos(α) * dl[2])
        for method in (bslLD.PolarSL(6), bslLD.PolarTaylor(6))
            a = rot(f0)
            bslLD.polar_kick!(a, g, dl[1], dl[2]; method = method)
            b = copy(f0)
            bslLD.polar_kick!(b, g, dlr[1], dlr[2]; method = method)
            @test maximum(abs, a .- rot(b)) < 1e-12
        end
    end

    @testset "translation of a Gaussian: convergence, pole crossing, mass" begin
        for (c, d) in (((0.7, 0.3), (0.4, -0.2)), ((0.3, 0.1), (-0.6, -0.2)))
            for method in (bslLD.PolarSL(6), bslLD.PolarTaylor(6))
                e32, _ = _ptranslate_err(32, 48, method; c = c, d = d)
                e64, dm = _ptranslate_err(64, 48, method; c = c, d = d)
                # (K2 W=6 crossing: 5.2 with the pole mass correction, 6.0 without)
                @test log2(e32 / e64) > 5.0
                method isa bslLD.PolarTaylor && @test dm < 1e-13
            end
            e8, _ = _ptranslate_err(64, 48, bslLD.PolarSL(8); c = c, d = d)
            @test e8 < 3e-9
        end
    end

    @testset "ragged Nφ(r) (K2, x advection, moments)" begin
        rule = bslLD.RaggedNphi(8; n0 = 20, nmin = 20, nmax = 48)
        @test rule(0.0) == 20 && rule(1.0) == 28 && rule(10.0) == 48 && iseven(rule(0.37))
        g = bslLD.PolarVGrid(0.0, 2pi / 1.5, 4, 64, rule; rmax = 6.0)
        @test !bslLD.uniform_nphi(g) && g.Nv == sum(g.nphi) && issorted(g.nphi)
        # init, density and x advection are per velocity node: same as uniform
        sp = bslLD.Distribution(g, 1e-3)
        x = g.xaxes[1]
        @test maximum(abs, bslLD.compute_density(sp, g).data .- (1 .+ 1e-3 .* cos.(1.5 .* x))) < 1e-7
        st = bslLD.SimulationTime(0.3, 1.0; phase = 0.4)
        bslLD.advectX!(sp, g, st)
        ex = [(1 + 1e-3 * cos(1.5 * (x[ix] - 0.3 * g.vr[iv] * cos(g.vphi[iv] - 0.4)))) * exp(-g.vr[iv]^2 / 2) / 2pi for ix in eachindex(x), iv = 1:g.Nv]
        @test maximum(abs, sp.data .- ex) < 1e-14
        # K2 translation on ragged rings; mass to round-off; K1 refuses ragged grids
        fun(vx, vy) = _pgauss(vx, vy, (0.7, 0.3), 0.6)
        f = [fun(_pvx(g, iv), _pvy(g, iv)) for ix = 1:1, iv = 1:g.Nv]
        m0 = sum(f[1, :] .* g.weights)
        bslLD.polar_kick!(f, g, 0.4, -0.2; method = bslLD.PolarTaylor(8))
        exk = [fun(_pvx(g, iv) - 0.4, _pvy(g, iv) + 0.2) for iv = 1:g.Nv]
        @test maximum(abs, f[1, :] .- exk) < 2e-8
        @test abs(sum(f[1, :] .* g.weights) / m0 - 1) < 1e-14
        @test_throws ArgumentError bslLD.polar_kick!(copy(f), g, 0.1, 0.0; method = bslLD.PolarSL(6))
        # the inner-ring extrapolation of not-carried modes beats zeros
        g2 = bslLD.PolarVGrid(0.0, 1.0, 1, 64, bslLD.RaggedNphi(12; n0 = 12, nmin = 12, nmax = 48); rmax = 6.0)
        errs = map((0, 2)) do e
            h = [exp(-_pvx(g2, iv)^2 / 2 - _pvy(g2, iv)^2 / 2) * cos(3 * _pvy(g2, iv)) for ix = 1:1, iv = 1:g2.Nv]
            bslLD.polar_kick!(h, g2, 0.02, 0.01; method = bslLD.PolarTaylor(8; extrap = e))
            hx = [exp(-(_pvx(g2, iv) - 0.02)^2 / 2 - (_pvy(g2, iv) - 0.01)^2 / 2) * cos(3 * (_pvy(g2, iv) - 0.01)) for iv = 1:g2.Nv]
            maximum(abs, h[1, :] .- hx)
        end
        @test errs[2] < errs[1] / 3
        # a per-ring vector equal to a uniform Nφ gives bit-identical K2 results
        gu = bslLD.PolarVGrid(0.0, 1.0, 1, 16, 24; rmax = 6.0)
        gv = bslLD.PolarVGrid(0.0, 1.0, 1, 16, fill(24, 16); rmax = 6.0)
        a = [fun(_pvx(gu, iv), _pvy(gu, iv)) for ix = 1:1, iv = 1:gu.Nv]
        b = copy(a)
        bslLD.polar_kick!(a, gu, 0.2, 0.1; method = bslLD.PolarTaylor(6))
        bslLD.polar_kick!(b, gv, 0.2, 0.1; method = bslLD.PolarTaylor(6))
        @test a == b
    end

    @testset "K3 spectral radial derivative (PolarSpectral)" begin
        K3 = bslLD.PolarSpectral()
        # exponential convergence, pole crossing, mass to round-off
        e12, _ = _ptranslate_err(12, 32, K3; c = (0.3, 0.1), d = (-0.6, -0.2))
        e16, dm = _ptranslate_err(16, 32, K3; c = (0.3, 0.1), d = (-0.6, -0.2))
        @test e16 < 1e-7 && e12 / e16 > 30
        @test dm < 1e-14
        e16o, _ = _ptranslate_err(16, 48, K3; c = (0.7, 0.3), d = (0.4, -0.2))
        e16k2, _ = _ptranslate_err(16, 48, bslLD.PolarTaylor(8); c = (0.7, 0.3), d = (0.4, -0.2))
        @test e16o < 1e-6 && e16o < e16k2 / 100
        # spectral derivative of an even (m = 0) and an odd (m = 1) mode
        g = bslLD.PolarVGrid(0.0, 1.0, 1, 24, 8; rmax = 7.0)
        st = bslLD._kick_static(g, Float64, K3)
        sp = bslLD._SpecDeriv(g, Float64, 2)
        U = zeros(ComplexF64, 24, 2)
        U[:, 1] .= exp.(-g.r .^ 2 ./ 2)
        U[:, 2] .= (1 + 2im) .* g.r .* exp.(-g.r .^ 2 ./ 2)
        D = similar(U)
        bslLD._k2_ddr!(D, U, st, sp)
        @test maximum(abs, D[:, 1] .+ g.r .* exp.(-g.r .^ 2 ./ 2)) < 1e-9
        @test maximum(abs, D[:, 2] .- (1 + 2im) .* (1 .- g.r .^ 2) .* exp.(-g.r .^ 2 ./ 2)) < 1e-9
        # no-op, rotation equivariance, Cartesian kick convention
        g = bslLD.PolarVGrid(0.0, 1.0, 3, 16, 24; rmax = 6.0)
        f0 = [_pgauss(_pvx(g, iv), _pvy(g, iv), (0.4, 0.2), 0.7) * (1 + 0.1ix) for ix = 1:3, iv = 1:g.Nv]
        f = copy(f0)
        bslLD.polar_kick!(f, g, 0.0, 0.0; method = K3)
        @test f == f0
        n, dl = 5, (0.3, -0.15)
        α = 2pi * n / 24
        rot(f) = (fr = similar(f); for j = 1:16, l = 1:24
            fr[:, g.offset[j]+l] = f[:, g.offset[j]+mod1(l - n, 24)]
        end; fr)
        a = rot(f0)
        bslLD.polar_kick!(a, g, dl[1], dl[2]; method = K3)
        b = copy(f0)
        bslLD.polar_kick!(b, g, cos(α) * dl[1] + sin(α) * dl[2], -sin(α) * dl[1] + cos(α) * dl[2]; method = K3)
        @test maximum(abs, a .- rot(b)) < 1e-12
        th, tau, E = 0.9, 0.25, [0.8, -0.4]
        dlc = tau .* (cos(th) * E[1] - sin(th) * E[2], sin(th) * E[1] + cos(th) * E[2])
        e = bslLD.VectorField([fill(E[1], 2), fill(E[2], 2), zeros(2)])
        gk = bslLD.PolarVGrid(0.0, 1.0, 2, 24, 48; rmax = 6.0)
        spk = bslLD.Distribution(gk, 0.0; initFuncx = x -> 1.0, initFuncv = (r, φ) -> _pgauss(r * cos(φ), r * sin(φ), (0.5, -0.3), 0.6))
        bslLD.advectV!(spk, gk, bslLD.SimulationTime(tau, 1.0; phase = th), e; method = K3)
        ex = [_pgauss(_pvx(gk, iv) - dlc[1], _pvy(gk, iv) - dlc[2], (0.5, -0.3), 0.6) for ix = 1:2, iv = 1:gk.Nv]
        @test maximum(abs, spk.data .- ex) < 1e-9
        @test_throws ArgumentError bslLD.PolarSpectral(; massfix = :bogus)
    end

    @testset "review fixes: edge outflow, Taylor orders, small Ns, even ragged rings" begin
        # mass at the outer edge (review case): the fix must not move the outflow to the pole
        fun(vx, vy) = _pgauss(vx, vy, (4.8, 0.0), 0.6)
        g = bslLD.PolarVGrid(0.0, 1.0, 1, 48, 48; rmax = 6.0)
        cfgs = ((:none, :wall), (:pole, :wall), (:profile, :wall), (:pole, :outflow))
        res = map(cfgs) do (mf, ed)
            bslLD.polar_mass_diagnostics(g; reset = true)
            f = [fun(_pvx(g, iv), _pvy(g, iv)) for ix = 1:1, iv = 1:g.Nv]
            m0 = sum(f[1, :] .* g.weights)
            bslLD.polar_kick!(f, g, 1.0, 0.0; method = bslLD.PolarTaylor(8; massfix = mf, edge = ed))
            ex = [fun(_pvx(g, iv) - 1, _pvy(g, iv)) for iv = 1:g.Nv]
            err = abs.(f[1, :] .- ex)
            d = bslLD.polar_mass_diagnostics(g)
            (maximum(err[g.vr .< 1]), maximum(err[g.vr .< 2]), sum(f[1, :] .* g.weights) / m0, d.outflow / m0, d.pole_correction / m0)
        end
        for r in res
            @test r[1] < 5e-6 && r[2] < 1e-3                 # no pole damage
            @test abs(r[3] - 1 - r[4]) < 1e-12                 # diagnostic = mass change
            @test r[5] < 1e-8                                  # pole defect is tiny
        end
        @test all(r -> abs(r[1] / res[1][1] - 1) < 0.1, res)   # the fix does not touch the pole
        @test abs(res[2][3] - 1) < 1e-13 && abs(res[3][3] - 1) < 1e-13   # :wall keeps mass
        @test res[1][3] < 0.75 && res[4][3] < 0.75             # :none / :outflow let it out
        # without edge mass, the fixed kicks conserve mass to round-off
        for m in (bslLD.PolarTaylor(8), bslLD.PolarSpectral())
            _, dm = _ptranslate_err(32, 48, m; c = (0.3, 0.1), d = (-0.6, -0.2))
            @test dm < 1e-14
        end
        # Taylor orders: only those stable on the imaginary axis, cfl within their limit
        for o in (1, 2, 5, 6)
            @test_throws ArgumentError bslLD.PolarTaylor(8; order = o)
            @test_throws ArgumentError bslLD.PolarSpectral(; order = o)
        end
        @test_throws ArgumentError bslLD.PolarTaylor(8; order = 4, cfl = 3.0)
        @test_throws ArgumentError bslLD.PolarSpectral(; order = 7, cfl = 2.0)
        for o in (3, 7)
            e, dm = _ptranslate_err(32, 48, bslLD.PolarSpectral(; order = o); c = (0.3, 0.1), d = (-0.6, -0.2))
            @test e < 1e-8 && dm < 1e-14     # order 3 has a 7e-9 time error here
        end
        # Ns smaller than the stencil half width / width (ghosts beyond the last ring)
        gs = bslLD.PolarVGrid(0.0, 1.0, 1, 3, 8; rmax = 4.0)
        f0 = [_pgauss(_pvx(gs, iv), _pvy(gs, iv), (0.2, 0.1), 1.0) for ix = 1:1, iv = 1:gs.Nv]
        for m in (bslLD.PolarTaylor(8), bslLD.PolarSL(8), bslLD.PolarSpectral())
            f = copy(f0)
            bslLD.polar_kick!(f, gs, 0.05, 0.02; method = m)
            ex = [_pgauss(_pvx(gs, iv) - 0.05, _pvy(gs, iv) - 0.02, (0.2, 0.1), 1.0) for iv = 1:gs.Nv]
            @test all(isfinite, f) && maximum(abs, f[1, :] .- ex) < 0.05
        end
        # ragged rule returns even ring sizes also for odd nmin/nmax
        rule = bslLD.RaggedNphi(5; n0 = 3, nmin = 9, nmax = 33)
        @test all(iseven, rule.(0:0.1:10)) && rule(0.0) == 10 && rule(10.0) == 32
        # quad_order = 0: the pole profile is a bump over the inner rings (no ring-K special case)
        g0 = bslLD.PolarVGrid(0.0, 1.0, 1, 16, 16; rmax = 5.0, quad_order = 0)
        h0 = [_pgauss(_pvx(g0, iv), _pvy(g0, iv), (0.2, 0.1), 0.8) for ix = 1:1, iv = 1:g0.Nv]
        hm = sum(h0[1, :] .* g0.weights)
        bslLD.polar_kick!(h0, g0, 0.1, 0.0; method = bslLD.PolarTaylor(4))
        @test abs(sum(h0[1, :] .* g0.weights) / hm - 1) < 1e-14
    end

    @testset "K2 mass fix variants" begin
        g = bslLD.PolarVGrid(0.0, 1.0, 1, 32, 48; rmax = 6.0)
        fun(vx, vy) = _pgauss(vx, vy, (0.3, 0.1), 0.6)
        res = map((:none, :profile, :pole)) do mf
            f = [fun(_pvx(g, iv), _pvy(g, iv)) for ix = 1:1, iv = 1:g.Nv]
            m0 = sum(f[1, :] .* g.weights)
            bslLD.polar_kick!(f, g, -0.6, -0.2; method = bslLD.PolarTaylor(8; massfix = mf))
            ex = [fun(_pvx(g, iv) + 0.6, _pvy(g, iv) + 0.2) for iv = 1:g.Nv]
            (maximum(abs, f[1, :] .- ex), abs(sum(f[1, :] .* g.weights) / m0 - 1))
        end
        @test res[1][2] > 1e-13                       # unfixed: truncation-size drift
        @test res[2][2] < 1e-14 && res[3][2] < 1e-14  # fixed: round-off
        @test maximum(r -> r[1], res) < 1.1 * minimum(r -> r[1], res)  # no accuracy cost
        @test_throws ArgumentError bslLD.PolarTaylor(8; massfix = :bogus)
    end

    @testset "Strang step through the Cartesian run-loop API" begin
        k = 1.5
        g = bslLD.PolarVGrid(0.0, 2pi / k, 8, 12, 16; rmax = 6.0)
        for method in (bslLD.PolarSL(6), bslLD.PolarTaylor(6))
            sp = bslLD.Distribution(g, 1e-3)
            st = bslLD.SimulationTime(0.05, 0.2)
            n0 = sum(bslLD.compute_density(sp, g).data)
            while bslLD.continue_advection(st)
                Ω = bslLD.gyro_frequency(sp, g)
                ph = st.phase
                sol = bslLD.solve_fields(bslLD.Moments(bslLD.compute_density(sp, g)), g, bslLD.PoissonSolver(-0.001))
                st.fraction_dt = 0.5
                bslLD.advectV!(sp, g, st, sol.E; method = method)
                st.phase = ph + Ω * st.dt / 2
                st.fraction_dt = 1.0
                bslLD.advectX!(sp, g, st)
                st.phase = ph + Ω * st.dt
                sol = bslLD.solve_fields(bslLD.Moments(bslLD.compute_density(sp, g)), g, bslLD.PoissonSolver(-0.001))
                st.fraction_dt = 0.5
                bslLD.advectV!(sp, g, st, sol.E; method = method)
                st.fraction_dt = 1.0
                st.phase = ph
                bslLD.advance!(st, Ω)
            end
            n1 = sum(bslLD.compute_density(sp, g).data)
            @test all(isfinite, sp.data)
            # |E| ≈ 0.7 here (ω_p² = 1000): K1 drifts at truncation level, K2 is exact
            @test abs(n1 / n0 - 1) < (method isa bslLD.PolarTaylor ? 1e-13 : 1e-5)
        end
    end
end
