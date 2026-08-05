function use_cpu!()
    set_execution_space!(; alloc = identity, exec = KernelAbstractions.CPU())
    return nothing
end

function cuda_available()
    return _cuda_available()
end

function use_cuda!()
    cuda_available() || error("CUDA is installed but no functional GPU is available")
    return _set_cuda_execution_space!()
end

function amdgpu_available()
    return _amdgpu_available()
end

function use_amdgpu!()
    amdgpu_available() || error("AMDGPU is installed but no functional GPU is available")
    return _set_amdgpu_execution_space!()
end

function backend_copy(x::AbstractArray)
    return bslLD.backend_array(Array(x))
end

function backend_copy(f::DistributionGrid)
    data = backend_copy(f.data)
    _, NX, NV, NXNV, ID = typeof(f).parameters[1:5]
    DT = eltype(data)
    return DistributionGrid{DT,NX,NV,NXNV,ID,typeof(data)}(data, DT(f.m), DT(f.q))
end

function backend_copy(e::VectorField)
    return VectorField([backend_copy(component) for component in e])
end

function backend_copy(e::ScalarField)
    return ScalarField(backend_copy(e.data))
end

function backend_copy(m::TensorField{DT,N,AT,2,NF}) where {DT,N,AT,NF}
    NS = isqrt(NF)
    return MatrixField([backend_copy(m[i, j]) for i = 1:NS, j = 1:NS])
end


function backend_synchronize!()
    exec = backend()
    KernelAbstractions.synchronize(exec)
    _backend_synchronize!(exec)
    return nothing
end
