module bslLD

using AbstractFFTs, Adapt, FFTW, KernelAbstractions
using Dierckx, Base.Threads, StaticArrays, ProgressMeter

const DEFAULT_ALLOCATOR = Ref{Function}(identity)
const DEFAULT_BACKEND = Ref{Any}(KernelAbstractions.CPU())
const CUDA_AVAILABLE_HOOK = Ref{Function}(() -> false)
const SET_CUDA_EXECUTION_SPACE_HOOK = Ref{Function}(
    () -> error(
        "CUDA-dependent functionality requires `using CUDA` in the active Julia session.",
    ),
)

function _allocator_ref()
    if !isdefined(@__MODULE__, :DEFAULT_ALLOCATOR)
        @eval const DEFAULT_ALLOCATOR = Ref{Function}(identity)
    end
    return getfield(@__MODULE__, :DEFAULT_ALLOCATOR)
end

function _backend_ref()
    if !isdefined(@__MODULE__, :DEFAULT_BACKEND)
        @eval const DEFAULT_BACKEND = Ref{Any}(KernelAbstractions.CPU())
    end
    return getfield(@__MODULE__, :DEFAULT_BACKEND)
end

allocate(x) = _allocator_ref()[](x)
backend() = _backend_ref()[]

_backend_array_matches(::Any, ::AbstractArray) = false
_cuda_available() = CUDA_AVAILABLE_HOOK[]()
_set_cuda_execution_space!() = SET_CUDA_EXECUTION_SPACE_HOOK[]()

_backend_synchronize!(::Any) = nothing

function backend_array(x::AbstractArray)
    exec = backend()
    if exec isa KernelAbstractions.CPU
        return x isa Array ? x : Array(x)
    end
    if _backend_array_matches(exec, x)
        return x
    end
    return allocate(x)
end

function set_execution_space!(alloc, exec)
    _allocator_ref()[] = alloc
    _backend_ref()[] = exec
    return nothing
end

function set_execution_space!(; alloc = nothing, exec = nothing)
    alloc === nothing || (_allocator_ref()[] = alloc)
    exec === nothing || (_backend_ref()[] = exec)
    return nothing
end

include("grid.jl")
include("time.jl")
include("distribution.jl")
include("fields.jl")
include("indexing.jl")
include("advectorCart.jl")
include("differential_operators.jl")
include("spectral.jl")
include("field_solver.jl")
include("solvers_electrostatic.jl")
include("solvers_vacuum.jl")
include("solvers_hybrid.jl")
include("advectorPolar.jl")
include("execution.jl")
include("sources.jl")

greet() = print("Hello World!")

end # module bslLD
