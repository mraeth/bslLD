#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__; io = devnull)

using Statistics

# Detects the GPU (is_cuda, ...), loads its package and bslLD, so the CPU benchmark also runs
# without a GPU.
include("../scripts/select_backend.jl")

const DEFAULTS = (nx = 1024, nv = 1024, samples = 50, warmup = 5, dt = 0.05, epsilon = 0.01)

# Each routine receives (f, grid, e, simTime, plan). The plain variants look the plan up in
# bslLD's cache, which is keyed on the identity of f.data and is therefore rebuilt for every
# fresh clone; the (plan) variants reuse one prebuilt plan and time the advection alone.
const ROUTINES = (
    ("advectX!      ", (f, grid, _, st, _plan) -> bslLD.advectX!(f, grid, st)),
    ("advectV!      ", (f, grid, e, st, _plan) -> bslLD.advectV!(f, grid, st, e)),
    (
        "advectX!(plan)",
        (f, grid, _, st, plan) -> for dir = 1:length(grid.xaxes)
            bslLD._advect_x_dir!(f, grid, st, dir, plan)
        end,
    ),
    (
        "advectV!(plan)",
        (f, grid, e, st, plan) -> for dir = 1:length(grid.vaxes)
            bslLD._advect_v_dir!(f, grid, st, e, dir, plan)
        end,
    ),
)

function parse_args(args)
    opts = Dict(string(key) => value for (key, value) in pairs(DEFAULTS))
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("Unexpected argument: $arg")
        key = arg[3:end]
        haskey(opts, key) || error("Unknown option --$key")
        i == length(args) && error("Missing value for --$key")
        value = args[i+1]
        opts[key] = opts[key] isa Int ? parse(Int, value) : parse(Float64, value)
        i += 2
    end
    return opts
end

function make_inputs(nx, nv, dt, epsilon)
    # 4th Grid argument is the number of spatial dimensions (1 for 1D1V).
    grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [nx, nv], 1)
    f = bslLD.Distribution(grid, epsilon)
    ex = @. 0.05 * sin(2pi * grid.xaxes[1] / grid.max[1])
    e = bslLD.VectorField([collect(ex)])
    simTime = bslLD.SimulationTime(dt, 10.0 * dt)
    plan = bslLD.AdvectionPlan(f.dist, grid)
    return grid, f, e, simTime, plan
end

function clone_inputs(f, e)
    return bslLD.backend_copy(f), bslLD.backend_copy(e)
end

function summarize(times)
    return (minimum = minimum(times), median = median(times), mean = mean(times))
end

function print_summary(device, routine, stats)
    println(
        rpad(device, 8),
        rpad(routine, 18),
        "min=",
        round(stats.minimum * 1e3; digits = 3),
        " ms  ",
        "median=",
        round(stats.median * 1e3; digits = 3),
        " ms  ",
        "mean=",
        round(stats.mean * 1e3; digits = 3),
        " ms",
    )
end

function benchmark_routine!(
    routine!,
    f_template,
    grid,
    e_template,
    simTime,
    plan,
    samples,
    warmup;
    sync! = (() -> nothing),
)
    for _ = 1:warmup
        f, e = clone_inputs(f_template, e_template)
        routine!(f, grid, e, simTime, plan)
        sync!()
    end

    times = Float64[]
    for _ = 1:samples
        f, e = clone_inputs(f_template, e_template)
        elapsed = @elapsed begin
            routine!(f, grid, e, simTime, plan)
            sync!()
        end
        push!(times, elapsed)
    end
    return summarize(times)
end

function bench(device, use_backend!, make_inputs, samples, warmup; sync! = (() -> nothing))
    use_backend!()
    grid, f_template, e_template, simTime, plan = make_inputs()

    println(device, " benchmarks")
    stats = Dict(
        label => benchmark_routine!(
            routine!,
            f_template,
            grid,
            e_template,
            simTime,
            plan,
            samples,
            warmup;
            sync!,
        ) for (label, routine!) in ROUTINES
    )
    for (label, _) in ROUTINES
        print_summary(device, label, stats[label])
    end
    return stats
end

function print_speedups(cpu_stats, gpu_stats)
    println("\nGPU speedup vs CPU (median runtime)")
    for (label, _) in ROUTINES
        speedup = cpu_stats[label].median / gpu_stats[label].median
        println(rpad(label, 18), round(speedup; digits = 2), "x")
    end
end

function main(args)
    opts = parse_args(args)
    make_case() = make_inputs(opts["nx"], opts["nv"], opts["dt"], opts["epsilon"])

    println("Advection benchmark configuration")
    println(
        "nx=$(opts["nx"]) nv=$(opts["nv"]) dt=$(opts["dt"]) epsilon=$(opts["epsilon"]) samples=$(opts["samples"]) warmup=$(opts["warmup"])",
    )

    cpu_stats = bench("CPU", bslLD.use_cpu!, make_case, opts["samples"], opts["warmup"])

    if is_cuda && bslLD.cuda_available()
        println()
        gpu_stats = bench(
            "GPU",
            bslLD.use_cuda!,
            make_case,
            opts["samples"],
            opts["warmup"];
            sync! = bslLD.backend_synchronize!,
        )
        print_speedups(cpu_stats, gpu_stats)
    else
        println("\nGPU benchmarks skipped.")
    end
end

Base.invokelatest(main, ARGS)
