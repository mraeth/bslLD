#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, "."); io = devnull)
Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)

using Statistics

is_cuda   = try success(`nvidia-smi`) catch; false end
is_amdgpu = !is_cuda && try success(`rocm-smi`) catch; false end
is_metal  = !is_cuda && !is_amdgpu && Sys.isapple() && Sys.ARCH === :aarch64
has_gpu   = is_cuda || is_amdgpu || is_metal

if is_cuda
    using CUDA
elseif is_amdgpu
    using AMDGPU
elseif is_metal
    using Metal
end

using bslLD

const DEFAULTS = (nx = 256, nv = 256, samples = 30, warmup = 5, dt = 0.05, epsilon = 0.01)

# Each routine receives (f, grid, e, simTime).
const ROUTINES = (
    ("advectX!", (f, grid, e, st) -> bslLD.advectX!(f, grid, st)),
    ("advectV!", (f, grid, e, st) -> bslLD.advectV!(f, grid, st, e)),
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
    # 4th Grid arg is number of spatial dimensions (1 for 1D1V).
    # Metal backend allocator auto-converts Float64 arrays to Float32 on upload.
    grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [nx, nv], 1)
    f = bslLD.Distribution(grid, epsilon)
    ex = @. 0.05 * sin(2π * grid.xaxes[1] / grid.max[1])
    e = bslLD.VectorField([collect(ex)])
    simTime = bslLD.SimulationTime(dt, 10.0 * dt)
    return grid, f, e, simTime
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
        rpad(routine, 12),
        "min=",    round(stats.minimum * 1e3; digits = 3), " ms  ",
        "median=", round(stats.median  * 1e3; digits = 3), " ms  ",
        "mean=",   round(stats.mean    * 1e3; digits = 3), " ms",
    )
end

function benchmark_routine!(
    routine!,
    f_template,
    grid,
    e_template,
    simTime,
    samples,
    warmup;
    sync! = () -> nothing,
)
    for _ in 1:warmup
        f, e = clone_inputs(f_template, e_template)
        routine!(f, grid, e, simTime)
        sync!()
    end
    times = Float64[]
    for _ in 1:samples
        f, e = clone_inputs(f_template, e_template)
        elapsed = @elapsed begin
            routine!(f, grid, e, simTime)
            sync!()
        end
        push!(times, elapsed)
    end
    return summarize(times)
end

function bench(device, use_backend!, make_inputs_fn, samples, warmup; sync! = () -> nothing)
    use_backend!()
    grid, f_template, e_template, simTime = make_inputs_fn()
    println("\n$device benchmarks")
    stats = Dict(
        label => benchmark_routine!(
            routine!, f_template, grid, e_template, simTime, samples, warmup; sync!,
        ) for (label, routine!) in ROUTINES
    )
    for (label, _) in ROUTINES
        print_summary(device, label, stats[label])
    end
    return stats
end

function print_speedups(cpu_stats, gpu_stats, gpu_label)
    println("\n$gpu_label speedup vs CPU (median runtime)")
    for (label, _) in ROUTINES
        speedup = cpu_stats[label].median / gpu_stats[label].median
        println(rpad(label, 12), round(speedup; digits = 2), "x")
    end
end

gpu_label() = is_cuda ? "CUDA" : is_amdgpu ? "AMDGPU" : is_metal ? "Metal" : "GPU"

function use_gpu!()
    is_cuda   && return bslLD.use_cuda!()
    is_amdgpu && return bslLD.use_amdgpu!()
    is_metal  && return bslLD.use_metal!()
end

function main(args)
    opts = parse_args(args)
    make_case() = make_inputs(opts["nx"], opts["nv"], opts["dt"], opts["epsilon"])

    println("CPU vs GPU comparison")
    println("nx=$(opts["nx"])  nv=$(opts["nv"])  dt=$(opts["dt"])  samples=$(opts["samples"])  warmup=$(opts["warmup"])")
    has_gpu && println("GPU backend: $(gpu_label())")

    cpu_stats = bench("CPU", bslLD.use_cpu!, make_case, opts["samples"], opts["warmup"])

    if has_gpu
        gpu_stats = bench(
            gpu_label(),
            use_gpu!,
            make_case,
            opts["samples"],
            opts["warmup"];
            sync! = bslLD.backend_synchronize!,
        )
        print_speedups(cpu_stats, gpu_stats, gpu_label())
    else
        println("\nNo GPU detected — GPU benchmarks skipped.")
    end
end

Base.invokelatest(main, ARGS)
