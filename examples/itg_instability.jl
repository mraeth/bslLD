include("../scripts/select_backend.jl")
bslLD.greet()

using HDF5

function stepStrang!(f, grid, simTime)
    phase_start = simTime.phase
    Ω = simTime.gyro_frequency

    # K–V half-step at phase(t)
    rho = bslLD.compute_density(f, grid)
    sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.AdiabaticSolver())
    bslLD.add_kappaT!(f, grid, simTime.dt * 0.5, 0.1, sol.E)
    simTime.fraction_dt = 0.5
    bslLD.advectV!(f, grid, simTime, sol.E)

    # X full-step at phase(t + dt/2)
    simTime.phase = phase_start + Ω * simTime.dt * 0.5
    simTime.fraction_dt = 1.0
    bslLD.advectX!(f, grid, simTime)

    # V–K half-step at phase(t + dt)
    simTime.phase = phase_start + Ω * simTime.dt
    rho = bslLD.compute_density(f, grid)
    sol = bslLD.solve_fields(bslLD.Moments(rho), grid, bslLD.AdiabaticSolver())
    simTime.fraction_dt = 0.5
    bslLD.advectV!(f, grid, simTime, sol.E)
    bslLD.add_kappaT!(f, grid, simTime.dt * 0.5, 0.1, sol.E)
    simTime.fraction_dt = 1.0

    simTime.phase = phase_start

    diags!(bslLD.compute_density(f, grid), simTime)
end

function write_scalarfield(sf::bslLD.ScalarField, filename::AbstractString, field::AbstractString)
    # Ensure parent directory exists
    dir = dirname(filename)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end

    data = Array(sf.data)
    h5open(filename, "cw") do file
        if haskey(file, field)
            delete_object(file, field)
        end
        write(file, field, data)
    end
    return nothing
end

Lx = 2pi/0.8
Ly = Lx
Lz = 240*pi

vmax = 4.0

Nx = 64
Ny = 64
Nz = 8
Nv = 16

nDiag = 20

grid =  bslLD.Grid([0.0,0.0,0.0,-vmax,-vmax,-vmax],[Lx, Ly, Lz, vmax, vmax, vmax],[Nx, Ny, Nz, Nv, Nv, Nv],3, 1.0, 3)
simTime = bslLD.SimulationTime(0.1, 10000.0, gyro_frequency=1.0)

initFuncx(x) = 1+ 0.0001 * randn()

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running Simulation")

    f = bslLD.Distribution(grid, 0.0, initFuncx = initFuncx);

    function diags!(rho, simTime)
        simTime.step % nDiag == 0 || return
        write_scalarfield(rho, "output/rho.h5", "rho_$(simTime.step)")
    end

    while bslLD.continue_advection(simTime, true)
        stepStrang!(f, grid, simTime)
        bslLD.advance!(simTime)
    end

end