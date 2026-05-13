# bslLD

Low-dimensional Backward Semi-Lagrangian solver for testing numerical methods.

This package serves as a testing ground for numerical methods intended for [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d), providing a simplified 1D/2D configuration space with 2D velocity space implementation.

## Features

- **Grid systems**: Cartesian and polar coordinate support
- **Distribution functions**: 1D1v, 1D2v, and 2D2v configurations
- **Advection methods**: Fourier-based and spline interpolation
- **Field solvers**: FFT-based Poisson solver for electrostatic problems
- **Parallelization**: Threaded velocity space advection
- **Optional GPU acceleration**: KernelAbstractions-based implementations for GPU computing

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/bslLD")
```

Or add to your Julia environment:

```julia
] dev /path/to/bslLD
```

Optional functionality is provided through weak dependencies:

- `CUDA.jl` enables GPU execution
- `Plots.jl` enables the plotting helper functions

For example, in an environment where you want GPU support and plotting:

```julia
using Pkg
Pkg.add(["CUDA", "Plots"])
```

## Quick Start

### CPU-only usage

```julia
using bslLD

# Create a 1D1v grid
grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.02, 0.1], 0.05, 2000, 1)

# Initialize distribution with custom velocity profile
initFuncv(v) = v^2 * exp(-v^2 / 2) / sqrt(2*pi)
f = bslLD.Distribution(grid, 0.01, initFuncv=initFuncv)

# CPU is the default backend
bslLD.use_cpu!()
```

### GPU usage with CUDA

`CUDA` is not loaded by `bslLD` automatically. To use GPU functionality, load `CUDA` in the Julia session and then switch backends:

```julia
using CUDA
using bslLD

grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.02, 0.1], 0.05, 2000, 1)
initFuncv(v) = v^2 * exp(-v^2 / 2) / sqrt(2*pi)
f = bslLD.Distribution(grid, 0.01, initFuncv=initFuncv)

bslLD.use_cuda!()
println("Using GPU acceleration")
```

If `CUDA` has not been loaded, `bslLD.use_cuda!()` will throw an error explaining that `using CUDA` is required first.

### Plotting

Plotting support is also optional. Load `Plots` before calling `bslLD` plotting helpers:

```julia
using Plots
using bslLD

fig = bslLD.heatmap_fv(Array(f.data[:, :]), grid)
```

If `Plots` has not been loaded, `bslLD.heatmap_fv(...)` will throw an error explaining that `using Plots` is required first.

## Project Structure

```
bslLD/
├── src/           # Core modules
├── test/          # Unit tests
├── notebooks/     # Development and exploration notebooks
└── examples/      # Example scripts and demos
```

## GPU Capabilities

The bslLD package leverages KernelAbstractions.jl to provide GPU-compatible implementations of core algorithms. The advection functions in `src/advectorCart.jl` and `src/advectorPolar.jl` utilize kernel-based parallelization that can run efficiently on both CPU and GPU backends.

Key GPU-enabled features:
- KernelAbstractions-based kernels for velocity space advection
- GPU-compatible FFT operations
- Parallel computation of distribution function updates
- Explicit backend switching through `bslLD.use_cpu!()` and `bslLD.use_cuda!()`

To run on GPU hardware, ensure you have appropriate GPU drivers and CUDA toolkit installed, then use:
```julia
using CUDA
using bslLD

bslLD.use_cuda!()
```

## Backend Selection

The package supports multiple execution backends through KernelAbstractions.jl:

- **CPU backend** (default): Operations run on the CPU
- **GPU backend**: Operations run on NVIDIA GPUs via CUDA when `CUDA.jl` is loaded

Switch between backends explicitly:
```julia
using bslLD

# CPU execution
bslLD.use_cpu!()

# GPU execution
using CUDA
bslLD.use_cuda!()
```

The package does not switch to CUDA automatically. The CUDA extension is only available once `CUDA.jl` has been loaded in the current Julia session.

## Dependencies

- FFTW.jl - Fast Fourier transforms
- Dierckx.jl - Spline interpolation
- StaticArrays.jl - Performance optimization
- Plots.jl - Optional plotting support
- CUDA.jl - Optional GPU acceleration

## Related Projects

- **BSL6D** - Full 6D Boltzmann solver
- **vlasovSL.jl** - Previous implementation (deprecated)

## Author

Mario Raeth (mario.raeth@ipp.mpg.de)
