function use_cpu!()
    set_execution_space!(; alloc=identity, exec=KernelAbstractions.CPU())
    return nothing
end

function cuda_available()
    try
        return CUDA.functional()
    catch
        return false
    end
end

function use_cuda!()
    cuda_available() || error("CUDA is installed but no functional GPU is available")
    set_execution_space!(; alloc=x -> CUDA.CuArray(x), exec=CUDA.CUDABackend())
    return nothing
end

function backend_copy(x::AbstractArray)
    return bslLD.backend_array(Array(x))
end

function backend_copy(f::DistributionGrid)
    data = backend_copy(f.data)
    _, NX, NV, NXNV, ID = typeof(f).parameters[1:5]
    DT = eltype(data)
    return DistributionGrid{DT, NX, NV, NXNV, ID, typeof(data)}(data)
end

function backend_copy(e::VectorField)
    return VectorField([backend_copy(component) for component in e])
end

function backend_copy(e::ScalarField)
    return ScalarField(backend_copy(e.data))
end


function backend_synchronize!()
    exec = backend()
    KernelAbstractions.synchronize(exec)
    if exec isa CUDA.CUDABackend
        CUDA.synchronize()
    end
    return nothing
end
