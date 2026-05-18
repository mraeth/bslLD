module bslLD

using AbstractFFTs, FFTW, KernelAbstractions
using Dierckx, Base.Threads, StaticArrays

const DEFAULT_ALLOCATOR = Ref{Function}(identity)
const DEFAULT_BACKEND = Ref{Any}(KernelAbstractions.CPU())

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
_cuda_available() = false

function _set_cuda_execution_space!()
    error("CUDA-dependent functionality requires `using CUDA` in the active Julia session.")
end

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

function set_execution_space!(; alloc=nothing, exec=nothing)
    alloc === nothing || (_allocator_ref()[] = alloc)
    exec === nothing || (_backend_ref()[] = exec)
    return nothing
end

include("grid.jl")
include("distribution.jl")
include("fields.jl")
include("advectorCart.jl")
include("differential_operators.jl")
include("maxwell.jl")
include("advectorPolar.jl")

greet() = print("Hello World!")

end # module bslLD
