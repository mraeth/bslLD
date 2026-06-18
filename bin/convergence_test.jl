using Pkg
Pkg.activate(".")
Pkg.develop(path = "..")

isCuda = try
    success(`nvidia-smi`)
catch
    false
end
if isCuda
    println("CUDA is available, using GPU acceleration.")
    using CUDA
end

using bslLD
bslLD.greet()

if isCuda
    println("Setting backend to CUDA.")
    bslLD.use_cuda!()
else
    println("CUDA not available, using CPU.")
end

# 1D2V grid with 32x32x32 grid points
grid = bslLD.Grid([0.0, -4.0, -4.0], [20.0, 4.0, 4.0], [32, 32, 32], 1, 1.0, 3)


function stepLie!(f, grid, simTime)
    bslLD.advectX!(f, grid, simTime)
    rho = bslLD.compute_density(f, grid)
    sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.AdiabaticSolver())
    bslLD.advectV!(f, grid, simTime, sol.E)
end

function stepStrang!(f, grid, simTime)
    phase_start = simTime.phase
    Ω = simTime.gyro_frequency

    # V half-step at phase(t)
    sol = bslLD.solve_fields(
        bslLD.Moments(bslLD.compute_density(f, grid)),
        grid,
        bslLD.AdiabaticSolver(),
    )
    simTime.fraction_dt = 0.5
    bslLD.advectV!(f, grid, simTime, sol.E)

    # X full-step at phase(t + dt/2)
    simTime.phase = phase_start + Ω * simTime.dt * 0.5
    simTime.fraction_dt = 1.0
    bslLD.advectX!(f, grid, simTime)

    # V half-step at phase(t + dt)
    simTime.phase = phase_start + Ω * simTime.dt
    sol = bslLD.solve_fields(
        bslLD.Moments(bslLD.compute_density(f, grid)),
        grid,
        bslLD.AdiabaticSolver(),
    )
    simTime.fraction_dt = 0.5
    bslLD.advectV!(f, grid, simTime, sol.E)
    simTime.fraction_dt = 1.0

    # restore so advance!() applies the correct full-step increment
    simTime.phase = phase_start
end

function run_test(dt, grid, stepFunc)
    println("Running test with dt = ", dt, " and step function ", stepFunc)
    nmax = round(Int, 1.0 / dt)
    simTime = bslLD.SimulationTime(dt, 1.0; nmax = nmax, gyro_frequency = 1.0)
    f = bslLD.Distribution(grid, 0.0001)

    while bslLD.continue_advection(simTime)
        stepFunc(f, grid, simTime)
        bslLD.advance!(simTime)
    end

    return bslLD.compute_density(f, grid).data
end


dts = [0.1, 0.05, 0.025, 0.0125]

dt_ref = 0.001


function l2norm(ds)
    return sqrt(sum(@views ds .^ 2))
end

println("Lie splitting: computing reference solution with dt = ", dt_ref)
rho_ref_Lie = run_test(dt_ref, grid, stepLie!)
println("Lie splitting: computing test solutions...")
error_Lie = [l2norm(run_test(dt, grid, stepLie!) .- rho_ref_Lie) for dt in dts]

println("Strang splitting: computing reference solution with dt = ", dt_ref)
rho_ref_Strang = run_test(dt_ref, grid, stepStrang!)
println("Strang splitting: computing test solutions...")
error_Strang = [l2norm(run_test(dt, grid, stepStrang!) .- rho_ref_Strang) for dt in dts]

println("\nConvergence rates:")
rates_Lie =
    [log(error_Lie[i] / error_Lie[i+1]) / log(dts[i] / dts[i+1]) for i = 1:(length(dts)-1)]
rates_Strang = [
    log(error_Strang[i] / error_Strang[i+1]) / log(dts[i] / dts[i+1]) for
    i = 1:(length(dts)-1)
]

println("Lie splitting:    ", round.(rates_Lie, digits = 3))
println("Strang splitting: ", round.(rates_Strang, digits = 3))

using Plots

plot(dts, error_Lie, label = "Lie", xscale = :log10, yscale = :log10, marker = :o)
plot!(dts, error_Strang, label = "Strang", xscale = :log10, yscale = :log10, marker = :o)
plot!(
    dts,
    dts/maximum(dts) * maximum(error_Lie),
    label = "O(dt)",
    linestyle = :dash,
    color = :black,
)
plot!(
    dts,
    dts .^ 2/maximum(dts) .^ 2 * maximum(error_Strang),
    label = "O(dt^2)",
    linestyle = :dashdot,
    color = :black,
)
xlabel!("Time step size (dt)")
ylabel!("Error in density")
title!("Convergence of Lie vs Strang Splitting")

savefig("convergence.png")
