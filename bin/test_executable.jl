using Pkg
Pkg.activate(".")
Pkg.develop(path="..")

if success(`nvidia-smi`)
    using CUDA
end


using bslLD
bslLD.greet()

bslLD.use_cuda!()
