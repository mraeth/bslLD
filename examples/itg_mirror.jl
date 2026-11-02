# 6D ITG-like case with a mirror (reflecting) wall in x1, written as a template: copy the file and
# edit it. The step function and the time loop are plain code, and the splitting (what is advanced,
# in which order, at which time level) is yours. f starts at zero and is built by `deposit!` from a
# target function over a Gaussian ramp, as bsl6d does.
#
#   julia --project=. examples/itg_mirror.jl                    # full size (GPU recommended)
#   BSLLD_TINY=1 julia --project=. examples/itg_mirror.jl       # tiny grid, CPU, seconds (used in CI)
#
# Output: chunk files in `output/` (format: src/production/diagnostics.jl). The profile and target below
# are an illustration, not the bsl6d benchmark profile.

include(joinpath(@__DIR__, "..", "scripts", "select_backend.jl"))

using Statistics
import PlasmaCore
using PlasmaCore: Mirror, Periodic
using bslLD.Production: Diagnostic, write!, close!, gaussian_ramp, deposit!

function main()
    tiny = get(ENV, "BSLLD_TINY", "0") == "1"

    # ---- parameters -------------------------------------------------------------------------
    DT, TEND = tiny ? (0.3, 1.5) : (0.06, 15000.0)
    Ns = tiny ? (16, 8, 8, 8, 8, 8) : (64, 64, 16, 32, 32, 32)
    L1, L2, L3, vmax = 20.41592653589793, 20.41592653589793, 753.98, 6.0
    KAPPAT = 0.1
    RAMP_T, RAMP_STD = tiny ? (0.6, 0.15) : (30.0, 3.0)    # ramp window [0, RAMP_T]
    SEED = 1e-11                      # relative amplitude of the perturbation deposited with f
    ZONAL_DYNAMIC = false             # false: n_e,avg frozen after the ramp, as bsl6d
    ODIR = get(ENV, "BSLLD_OUTPUT", "output")

    # ---- grid, species, time: plain objects ---------------------------------------------------
    # x1 is a mirror axis with nodes on both walls: dx = L1/(N1-1), so etaMax = L1 + dx.
    grid = bslLD.Grid(
        [0.0, 0.0, 0.0, -vmax, -vmax, -vmax],
        [Ns[1] * L1 / (Ns[1] - 1), L2, L3, vmax, vmax, vmax],
        collect(Ns),
        3,
        1.0,
        3,
    )
    f = bslLD.Distribution(grid, 0.0; initFuncx = x -> 0.0)       # initial state: f = 0
    simTime = bslLD.SimulationTime(DT, TEND)
    Ω = bslLD.gyro_frequency(f, grid)
    MX, MV = bslLD.Lagrange(8), bslLD.Lagrange(6)

    # The mirror shift per x1 sweep should stay below one cell: sub-step the sweep (decision of the
    # script, not of the library). |v_x1| <= sqrt(2) vmax in the gyro-frame.
    NSUB1 = max(1, ceil(Int, sqrt(2) * vmax * DT / grid.delta[1]))

    # ---- profile and target: user code ----------------------------------------------------------
    fold(R) = (r = mod(R, 2L1); r > L1 ? 2L1 - r : r)           # reflect gyrocentres into [0, L1]
    Tprofile(R) = exp(-KAPPAT * (fold(R) - L1 / 2))
    device(a, dim) =
        bslLD.backend_array(reshape(collect(a), ntuple(d -> d == dim ? length(a) : 1, 6)))
    X = ntuple(d -> device(grid.xaxes[d], d), 3)
    U = ntuple(d -> device(grid.vaxes[d], 3 + d), 3)
    function target(grid, t)    # Maxwellian at the gyrocentre R1 = x1 + v2(t), perturbed in x2, x3
        R1 = @. X[1] + (-sin(t) * U[1] + cos(t) * U[2])
        T = Tprofile.(R1)
        v2 = @. U[1]^2 + U[2]^2 + U[3]^2
        pert = @. 1 + SEED * cos(2pi * X[2] / L2) * cos(2pi * X[3] / L3)
        return @. pert * exp(-v2 / (2 * T)) / (2pi * T)^(3 / 2)
    end
    ramp = gaussian_ramp(RAMP_T / 2, RAMP_STD)            # integral of the weight is 1

    # ---- fields: adiabatic solver; flux-surface average and filter are the script's choice -------
    solver = bslLD.AdiabaticSolver((Mirror(), Periodic(), Periodic()); order = 8)
    nfs = nothing                                               # n_e,avg(x1)
    function fields!(t)
        rho = bslLD.compute_density(f, grid)
        if nfs === nothing
            nfs = fill!(similar(rho.data, size(rho.data, 1), 1, 1), 0)
        end
        (t < RAMP_T || ZONAL_DYNAMIC) && (nfs .= mean(rho.data, dims = (2, 3)))
        rho.data .-= nfs
        sol = bslLD.solve_fields(bslLD.Moments(rho), grid, solver)
        for d = 1:3     # remove the three bins around Nyquist in x2 (periodic); x1 is not filtered
            PlasmaCore.fourier_filter!(
                sol.E[d],
                grid;
                remove = (2 => (Ns[2]÷2-1):(Ns[2]÷2+1),),
            )
        end
        return sol
    end

    # ---- diagnostics: one per field, explicit every / buffer ----------------------------------------
    diags = (
        Diagnostic(
            () -> bslLD.compute_density(f, grid),
            grid;
            dir = ODIR,
            name = "n",
            every = tiny ? 2 : 5,
            buffer = tiny ? 3 : 200,
            meta = Dict("description" => "ion density, raw, per unit volume"),
        ),
        Diagnostic(
            () -> bslLD.ScalarField(f.data),
            grid;
            dir = ODIR,
            name = "f",
            every = tiny ? 5 : 5000,
            buffer = 1,
            meta = Dict("description" => "ion distribution function"),
        ),
    )

    # ---- the step: the splitting is written here --------------------------------------------------
    function step!(simTime)
        t0, φ0 = simTime.current_T, simTime.phase
        deposit!(f, grid, target, t0; weight = ramp(t0), dt = DT / 2)
        simTime.phase = φ0 + Ω * DT / 2
        simTime.fraction_dt = 1 / NSUB1
        for _ = 1:NSUB1
            bslLD.advectX!(f, grid, simTime, 1; method = MX, boundary = Mirror())
        end
        simTime.fraction_dt = 1.0
        for d = 2:3
            bslLD.advectX!(f, grid, simTime, d; method = MX, boundary = Periodic())
        end
        sol = fields!(t0)
        deposit!(f, grid, target, t0 + DT / 2; weight = ramp(t0 + DT / 2), dt = DT / 2)
        simTime.phase = φ0 + Ω * DT
        bslLD.advectV!(f, grid, simTime, sol.E; method = MV)
        simTime.phase = φ0
    end

    # ---- the loop, written by the user -----------------------------------------------------------------
    sol = fields!(0.0)
    simTime.fraction_dt = 0.5
    bslLD.advectV!(f, grid, simTime, sol.E; method = MV)       # V/2 pre-step
    simTime.fraction_dt = 1.0
    while bslLD.continue_advection(simTime, !tiny)
        step!(simTime)
        bslLD.advance!(simTime, Ω)
        foreach(d -> write!(d, simTime), diags)
    end
    foreach(close!, diags)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
