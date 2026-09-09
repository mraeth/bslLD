module bslLD

using AbstractFFTs, Adapt, FFTW, KernelAbstractions
using Dierckx, Base.Threads, StaticArrays, ProgressMeter

using PlasmaCore:
    # Backend infrastructure
    DEFAULT_BACKEND, DEFAULT_ALLOCATOR,
    CUDA_AVAILABLE_HOOK, SET_CUDA_EXECUTION_SPACE_HOOK,
    AMDGPU_AVAILABLE_HOOK, SET_AMDGPU_EXECUTION_SPACE_HOOK,
    METAL_AVAILABLE_HOOK, SET_METAL_EXECUTION_SPACE_HOOK,
    allocate, backend, backend_array, set_execution_space!,
    _allocator_ref, _backend_ref,
    _backend_array_matches,
    _cuda_available, _set_cuda_execution_space!,
    _amdgpu_available, _set_amdgpu_execution_space!,
    _metal_available, _set_metal_execution_space!,
    _backend_synchronize!,
    # Grid
    Grid, Cart, Polar, CartGrid, PolarGrid, outer_product,
    # Time
    SimulationTime, advance!, continue_advection, elapsed_seconds, reset_timer!,
    # Indexing
    index_nd_to_1d, index_1d_to_nd, index_combined_to_1d, index_1d_to_combined,
    spectral_multiply_kernel!,
    # Fields
    TensorField, ScalarField, VectorField, MatrixField,
    empty_scalarfield, empty_vectorfield, empty_matrixfield,
    zero_vectorfield_like, zero_scalarfield_like,
    # Distribution data (physics constructors and moments live in bslLD kinetics files)
    DistributionGrid, DistributionGridImpl,
    DistributionGrid1d1v, DistributionGrid1d2v, DistributionGrid2d2v,
    # Spectral operators (public)
    SpectralWorkspace, DifferentiateContext,
    differentiate, grad, div, curl,
    spatial_ndims, ncomponents,
    fft_spatial, ifft_spatial, spatial_fft_dims,
    spectral_wavenumbers, spectral_wavenumber_squared, spectral_wavenumber_views,
    # Spectral operators (private, needed by solver files)
    _solver_workspace_cache, _solver_workspace_cache_lock,
    _spectral_ws_cache, _spectral_ws_cache_lock,
    _get_spectral_workspace, _spectral_ws_key,
    _differentiate_impl!, _spectral_curl_hat!,
    _fwd_fft_to!, _inv_fft_from!,
    _apply_div!, _apply_curl!,
    # Field solver interface
    AbstractFieldSolver, Moments, FieldSolution,
    vectorfield_from_spatial_components, zero_vectorfield3, background_field,
    # Execution utilities
    use_cpu!, cuda_available, use_cuda!, amdgpu_available, use_amdgpu!,
    metal_available, use_metal!, backend_copy, backend_synchronize!

include("kinetics/species.jl")
include("kinetics/advectorCart.jl")
include("kinetics/advectorPolar.jl")
include("kinetics/initialization.jl")
include("kinetics/moments.jl")
include("kinetics/moment_response.jl")
include("maxwell/solvers_electrostatic.jl")
include("maxwell/solvers_vacuum.jl")
include("maxwell/solvers_hybrid.jl")
include("maxwell/cold_plasma.jl")
include("kinetics/exbBracketCart.jl")
include("sources.jl")

end # module bslLD
