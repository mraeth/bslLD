using Pkg
Pkg.activate(".")
Pkg.develop(path = "..")

is_cuda = try
    success(`nvidia-smi`)
catch
    ;
    false
end
is_amdgpu = !is_cuda && try
    success(`rocm-smi`)
catch
    ;
    false
end
is_metal = !is_cuda && !is_amdgpu && Sys.isapple() && Sys.ARCH === :aarch64

if is_cuda
    println("NVIDIA GPU detected — loading CUDA.")
    using CUDA
elseif is_amdgpu
    println("AMD GPU detected — loading AMDGPU.")
    using AMDGPU
elseif is_metal
    println("Apple Silicon detected — loading Metal.")
    using Metal
else
    println("No GPU detected — using CPU.")
end

using bslLD

if is_cuda
    bslLD.use_cuda!()
elseif is_amdgpu
    bslLD.use_amdgpu!()
elseif is_metal
    bslLD.use_metal!()
else
    bslLD.use_cpu!()
end

println("Active backend: ", bslLD.backend())
