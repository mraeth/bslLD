using Pkg
Pkg.activate(".")
Pkg.develop(path="..")

isCuda = success(`nvidia-smi`)

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

#2D3V grid with 32x8 32x32x32 grid points 
grid = bslLD.Grid([0.0,0.0, -4.0,-4.0,-4.0],[20.0,20.0, 4.0,4.0,4.0],[32,32,33,33,33],2, 1.0, 2)
simTime = bslLD.SimulationTime(0.01, 1, gyro_frequency=1.0)

f = bslLD.Distribution(grid, 0.0001);

function step!(f, grid, simTime)
    bslLD.advectX!(f, grid, simTime)
    rho = bslLD.compute_density(f, grid)
    sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.AdiabaticSolver())
    bslLD.advectV!(f, grid, simTime, sol.E)

end

step!(f, grid, simTime)

function print_progress(simTime)
    println(round(simTime.current_T, sigdigits=5), "; ", simTime.step, "; ", round(bslLD.elapsed_seconds(simTime), sigdigits=5))

end

while bslLD.continue_advection(simTime)
    bslLD.reset_timer!(simTime)
    step!(f, grid, simTime)
    bslLD.advance!(simTime)
    print_progress(simTime)
end
