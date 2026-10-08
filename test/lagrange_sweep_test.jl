# The four Lagrange sweep kernels (:point, :line, :cached, :auto) must agree bit for bit.
using Test
import bslLD

@testset "Lagrange sweep kernels are bit-identical" begin
    modes = (:point, :line, :cached, :auto)
    old_mode, old_min = bslLD._SWEEP_MODE[], bslLD._AUTO_MIN_LINES[]

    # rough, non-smooth input: a bad stencil index would show up immediately
    function rough!(f, seed)
        for (k, i) in enumerate(eachindex(f.data))
            f.data[i] = 1 + 0.5 * sin(1.7 * k + seed) + 0.25 * cos(0.37 * k * seed)
        end
        return f
    end

    function run_sweeps(mode, grid, nv, W, bc; vsweep = true)
        bslLD._SWEEP_MODE[] = mode
        bslLD._AUTO_MIN_LINES[] = 4              # exercises both branches of :auto
        simTime = bslLD.SimulationTime(0.21, 0.21)
        f = rough!(bslLD.Distribution(grid, 0.0), 1.0)
        bslLD.advectX!(f, grid, simTime; method = bslLD.Lagrange(W), boundary = bc)
        if vsweep
            n = size(f.data)[1:length(grid.xaxes)]
            e = bslLD.VectorField([0.3 .+ 0.1 .* sin.(1:prod(n)) |> x -> reshape(x, n) for _ = 1:nv])
            bslLD.advectV!(f, grid, simTime, e; method = bslLD.Lagrange(W))
        end
        return copy(f.data)
    end

    try
        @testset "periodic, $(nx)D$(nv)V, W=$W" for (nx, nv) in ((1, 1), (1, 2), (2, 2)), W in (4, 8)
            eta_min = vcat(fill(0.0, nx), fill(-2.0, nv))
            eta_max = vcat(fill(2pi, nx), fill(2.0, nv))
            counts = vcat(fill(12, nx), fill(10, nv))
            grid = bslLD.Grid(eta_min, eta_max, counts, nx)
            ref = run_sweeps(:point, grid, nv, W, bslLD.Periodic())
            for mode in (:line, :cached, :auto)
                @test run_sweeps(mode, grid, nv, W, bslLD.Periodic()) == ref
            end
        end

        @testset "mirror in x, W=$W" for W in (4, 8)
            nx, nvx = 16, 6
            grid = bslLD.Grid([0.0, -1.5, -1.5], [2pi, 1.5, 1.5], [nx, nvx, nvx], 1)
            ref = run_sweeps(:point, grid, 2, W, bslLD.Mirror(); vsweep = false)
            for mode in (:line, :cached, :auto)
                @test run_sweeps(mode, grid, 2, W, bslLD.Mirror(); vsweep = false) == ref
            end
        end
    finally
        bslLD._SWEEP_MODE[], bslLD._AUTO_MIN_LINES[] = old_mode, old_min
    end
end
