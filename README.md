# bslLD

Low-dimensional Backward Semi-Lagrangian solver for testing numerical methods.

This package serves as a testing ground for numerical methods intended for [BSL6D](https://gitlab.mpcdf.mpg.de/bsl6d/bsl6d), providing a simplified 1D/2D configuration space with 2D velocity space implementation.

## Features

- **Grid systems**: Cartesian and polar coordinate support
- **Distribution functions**: 1D1v, 1D2v, and 2D2v configurations
- **Advection methods**: Fourier-based and spline interpolation
- **Field solvers**: FFT-based Poisson solver for electrostatic problems
- **Parallelization**: Threaded velocity space advection
- **Hardware abstraction**: Unified CPU/GPU execution with KernelAbstractions.jl
- **GPU acceleration**: Support for CUDA-enabled GPUs with automatic fallback to CPU

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
