# bslLD

Low-dimensional Backward Semi-Lagrangian solver for testing numerical methods.

This package serves as a testing ground for numerical methods intended for [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d), providing a simplified 1D/2D configuration space with 2D velocity space implementation.

## Features

- **Grid systems**: Cartesian and polar coordinate support
- **Distribution functions**: 1D1v, 1D2v, and 2D2v configurations
- **Advection methods**: Fourier-based and spline interpolation
- **Field solvers**: FFT-based Poisson solver for electrostatic problems
- **Parallelization**: Threaded velocity space advection
- **GPU acceleration**: KernelAbstractions-based implementations for GPU computing

## Installation

```julia
using Pkg
Pkg.develop(path="/path/to/bslLD")
```

Or add to your Julia environment:

```julia
] dev /path/to/bslLD
```

## Quick Start

```julia
using bslLD

# Create a 1D1v grid
grid = bslLD.Grid([0.0, -6.0], [10.0, 6.0], [0.02, 0.1], 0.05, 2000, 1)

# Initialize distribution with custom velocity profile
initFuncv(v) = v^2 * exp(-v^2 / 2) / sqrt(2*pi)
f = bslLD.Distribution(grid, 0.01, initFuncv=initFuncv)

# Switch to GPU (if available)
try
    bslLD.use_cuda!()
    println("Using GPU acceleration")
catch
    println("GPU not available, using CPU")
end

# Visualize
using Plots
heatmap(f.data[:, :]', c=:viridis)
```

## Project Structure

```
bslLD/
├── src/           # Core modules
├── test/          # Unit tests
├── notebooks/     # Development and exploration notebooks
└── examples/      # Example scripts and demos
```

## GPU Capabilities

The bslLD package leverages KernelAbstractions.jl to provide GPU-compatible implementations of core algorithms. The advection functions in `advector.jl` utilize kernel-based parallelization that can run efficiently on both CPU and GPU backends.

Key GPU-enabled features:
- KernelAbstractions-based kernels for velocity space advection
- GPU-compatible FFT operations
- Parallel computation of distribution function updates
- Automatic backend selection (CPU/GPU) through KernelAbstractions

To run on GPU hardware, ensure you have appropriate GPU drivers and CUDA toolkit installed, then use:
```julia
using CUDA
# All bslLD operations will automatically leverage GPU acceleration
```

## Backend Selection

The package supports multiple execution backends through KernelAbstractions.jl:

- **CPU backend** (default): Operations run on the CPU
- **GPU backend**: Operations run on NVIDIA GPUs via CUDA

Switch between backends by loading the appropriate package:
```julia
# For CPU execution (default)
# No special import needed

# For GPU execution
using CUDA
# All bslLD operations will automatically use GPU backend
```

The backend is automatically selected based on the loaded packages and available hardware. KernelAbstractions handles the backend dispatch transparently, allowing the same code to run efficiently on different hardware platforms.

## Dependencies

- FFTW.jl - Fast Fourier transforms
- Dierckx.jl - Spline interpolation
- StaticArrays.jl - Performance optimization
- Plots.jl - Visualization
- CUDA.jl - GPU acceleration (optional)

## Related Projects

- **BSL6D** - Full 6D Boltzmann solver
- **vlasovSL.jl** - Previous implementation (deprecated)

## Author

Mario Raeth (mario.raeth@ipp.mpg.de)
