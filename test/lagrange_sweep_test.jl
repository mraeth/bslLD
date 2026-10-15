# The tiled Lagrange sweeps and the general (one thread per point) sweep must agree bit for bit,
# and the general mirror sweep must match a plain-loop reference (to rounding: FMA on GPU).
using Test
import bslLD

@testset "Lagrange sweep kernels are bit-identical" begin
    # rough, non-smooth input: a bad stencil index would show up immediately
    function rough!(f, seed)
        for (k, i) in enumerate(eachindex(f.data))
            f.data[i] = 1 + 0.5 * sin(1.7 * k + seed) + 0.25 * cos(0.37 * k * seed)
        end
        return f
    end

    function run_sweeps(tiled, grid, nv, W, bc; vsweep = true)
        simTime = bslLD.SimulationTime(0.21, 0.21)
        sp = rough!(bslLD.Distribution(grid, 0.0), 1.0)
        plan = bslLD._get_plan(sp.dist, grid)
        method = bslLD.Lagrange(W)
        xdirs = 1:length(grid.xaxes)
        ctxs = [bslLD._x_context(sp, grid, simTime, d, plan) for d in xdirs]
        @test all(c -> bslLD._tiled(c, Val(W), bc, Float64), ctxs)   # the tiled path is exercised
        bslLD._advect!(sp.dist, plan, method, bc, ctxs, xdirs; tiled)
        if vsweep
            n = size(sp.data)[xdirs]
            e = bslLD.VectorField([
                0.3 .+ 0.1 .* sin.(1:prod(n)) |> x -> reshape(x, n) for _ = 1:nv
            ])
            vctxs = [bslLD._v_context(sp, grid, simTime, e, d, plan) for d = 1:nv]
            bslLD._advect!(sp.dist, plan, method, bc, vctxs, length(xdirs) .+ (1:nv); tiled)
        end
        return copy(sp.data)
    end

    @testset "periodic, $(nx)D$(nv)V, W=$W" for (nx, nv) in ((1, 1), (1, 2), (2, 2)),
        W in (4, 8)

        eta_min = vcat(fill(0.0, nx), fill(-2.0, nv))
        eta_max = vcat(fill(2pi, nx), fill(2.0, nv))
        counts = vcat(fill(12, nx), fill(10, nv))
        grid = bslLD.Grid(eta_min, eta_max, counts, nx)
        bc = bslLD.Periodic()
        @test run_sweeps(true, grid, nv, W, bc) == run_sweeps(false, grid, nv, W, bc)
    end

    @testset "mirror in x, W=$W" for W in (4, 8)
        nx, nvx = 16, 6
        grid = bslLD.Grid([0.0, -1.5, -1.5], [2pi, 1.5, 1.5], [nx, nvx, nvx], 1)
        bc = bslLD.Mirror()
        @test run_sweeps(true, grid, 2, W, bc; vsweep = false) ==
              run_sweeps(false, grid, 2, W, bc; vsweep = false)
    end

    # Mirror sweep in plain loops: node fold about 1 and n, reflected nodes taken from the
    # velocity-flipped partner line; same weights and summation order as the kernels.
    function mirror_reference(src, ctx, W)
        sx, sv = ctx.sizes_x, ctx.sizes_v
        n, dir = sx[ctx.dir], ctx.dir
        out = similar(src)
        for p in eachindex(src)
            ixs, ivs = bslLD.index_1d_to_combined(p, sx, sv)
            cells, alpha = bslLD.shift_split(bslLD._shift_cells(ctx, ixs, ivs))
            w = bslLD.lagrange_weights(Val(W), alpha, eltype(src))
            function val(i)
                j, v =
                    1 <= i <= n ? (i, ivs) :
                    (i < 1 ? min(2 - i, n) : max(2n - i, 1), bslLD._flip_v(ctx, ivs))
                src[bslLD.index_combined_to_1d(Base.setindex(ixs, j, dir), v, sx, sv)]
            end
            acc = zero(eltype(src))
            for m = 1:W
                acc += w[m] * val(ixs[dir] - cells - W ÷ 2 - 1 + m)
            end
            out[p] = acc
        end
        return out
    end

    @testset "general mirror sweep against a plain-loop reference, x$dir, W=$W" for dir in
                                                                                    (1, 2),
        W in (4, 8)
        # 2D2V, shifts up to ~5 cells so that stencils reach deep into the reflected halo
        grid = bslLD.Grid([0.0, 0.0, -1.5, -1.5], [2pi, 2pi, 1.5, 1.5], [14, 12, 6, 6], 2)
        sp = rough!(bslLD.Distribution(grid, 0.0), 2.0)
        plan = bslLD._get_plan(sp.dist, grid)
        ctx = bslLD._x_context(sp, grid, bslLD.SimulationTime(1.2, 1.2), dir, plan)
        @test bslLD._mirror_shift_bound(ctx) > 3
        src = copy(sp.data)
        dst = similar(src)
        bslLD._lagrange_sweep!(dst, src, ctx, Val(W), bslLD.Mirror(), plan, false)
        # host reference; on GPU backends FMA contraction changes the last bit, so compare with
        # a tolerance (a wrong halo node would give O(1) differences, the sweep changes f by ~1)
        ref = mirror_reference(Array(src), bslLD.Adapt.adapt(Array, ctx), W)   # host copies
        @test isapprox(Array(dst), ref; rtol = 1e-13)
    end

    @testset "tiles respect the 48 KiB shared-memory limit" begin
        # 32 lines of (n + W) Float64 values plus one Int cell shift each
        @test bslLD._tile_fits(Float64, 183, 8)
        @test !bslLD._tile_fits(Float64, 184, 8)
        @test bslLD._tile_fits(Float32, 360, 8)
    end
end
