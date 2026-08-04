using Pkg
Pkg.activate(".")
Pkg.develop(path = "..")

is_cuda   = try success(`nvidia-smi`) catch; false end
is_amdgpu = !is_cuda && try success(`rocm-smi`) catch; false end

if is_cuda
    println("NVIDIA GPU detected — loading CUDA.")
    using CUDA
elseif is_amdgpu
    println("AMD GPU detected — loading AMDGPU.")
    using AMDGPU
else
    println("No GPU detected — using CPU.")
end

using bslLD

if is_cuda
    bslLD.use_cuda!()
elseif is_amdgpu
    bslLD.use_amdgpu!()
else
    bslLD.use_cpu!()
end

println("Active backend: ", bslLD.backend())
