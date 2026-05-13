module bslLDCUDAExt

using bslLD
import CUDA

bslLD._backend_array_matches(exec::CUDA.CUDABackend, x::CUDA.CuArray) = true

function bslLD._cuda_available()
    try
        return CUDA.functional()
    catch
        return false
    end
end

function bslLD._set_cuda_execution_space!()
    bslLD.set_execution_space!(; alloc=x -> CUDA.CuArray(x), exec=CUDA.CUDABackend())
    return nothing
end

function bslLD._backend_synchronize!(exec::CUDA.CUDABackend)
    CUDA.synchronize()
    return nothing
end

end # module bslLDCUDAExt
