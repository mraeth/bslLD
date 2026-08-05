struct PoissonSolver{DT<:AbstractFloat} <: AbstractFieldSolver
    factor::DT
end
PoissonSolver() = PoissonSolver(1.0)
PoissonSolver(factor::Real) = PoissonSolver(float(factor))

struct AdiabaticSolver <: AbstractFieldSolver end

# ── AdiabaticSolver workspace ────────────────────────────────────────────────

struct AdiabaticSolverWorkspace{SW,EVF,ZVF,ZSF}
    sw::SW      # SpectralWorkspace (shared with other solvers on the same grid)
    E_vf::EVF   # pre-allocated VectorField written in-place each step
    zero_vf3::ZVF
    zero_phi::ZSF  # zero ScalarField used for the FieldSolution.phi slot
end

function _make_adiabatic_workspace(rho::ScalarField, grid::Grid)
    sw = _get_spectral_workspace(rho.data, grid)
    E_vf = VectorField([ScalarField(fill!(similar(rho.data, Float64), 0.0)) for _ = 1:3])
    zero_vf3 = zero_vectorfield3(grid)
    zero_phi = ScalarField(fill!(similar(rho.data, Float64), 0.0))
    return AdiabaticSolverWorkspace(sw, E_vf, zero_vf3, zero_phi)
end

function _get_adiabatic_workspace(rho::ScalarField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    key = (
        :adiabatic,
        size(rho.data),
        eltype(rho.data),
        ndims_x,
        Tuple(grid.delta[1:ndims_x]),
    )
    lock(_solver_workspace_cache_lock) do
        get!(() -> _make_adiabatic_workspace(rho, grid), _solver_workspace_cache, key)
    end
end

# ── PoissonSolver workspace ──────────────────────────────────────────────────

struct PoissonSolverWorkspace{SW,K2,PHI,EVF,ZVF}
    sw::SW
    k2::K2        # Float64 field-sized — spectral |k|²
    phi_buf::PHI  # Float64 field-sized — real phi
    E_vf::EVF
    zero_vf3::ZVF
end

function _make_poisson_workspace(rho::ScalarField, grid::Grid)
    sw = _get_spectral_workspace(rho.data, grid)
    ndims_x = spatial_ndims(grid)
    # Pre-compute |k|² on the backend
    k2 = fill!(similar(rho.data, Float64), 0.0)
    for d = 1:ndims_x
        kd = sw.k[d]
        # kd is 1-D; broadcast as a length-n_d slice along dim d
        k2 .+= reshape(kd, ntuple(i -> i == d ? length(kd) : 1, ndims(k2))) .^ 2
    end
    phi_buf = fill!(similar(rho.data, Float64), 0.0)
    E_vf = VectorField([ScalarField(fill!(similar(rho.data, Float64), 0.0)) for _ = 1:3])
    zero_vf3 = zero_vectorfield3(grid)
    return PoissonSolverWorkspace(sw, k2, phi_buf, E_vf, zero_vf3)
end

function _get_poisson_workspace(rho::ScalarField, grid::Grid, factor::AbstractFloat)
    ndims_x = spatial_ndims(grid)
    key = (
        :poisson,
        size(rho.data),
        eltype(rho.data),
        ndims_x,
        Tuple(grid.delta[1:ndims_x]),
        factor,
    )
    lock(_solver_workspace_cache_lock) do
        get!(() -> _make_poisson_workspace(rho, grid), _solver_workspace_cache, key)
    end
end

# ── Poisson helpers (allocating — kept for non-hot-path use) ─────────────────

function poisson_potential(
    rho::ScalarField,
    grid::Grid,
    coefficient::T,
) where {T<:AbstractFloat}
    rhohat = fft_spatial(rho.data, grid)
    k2 = spectral_wavenumber_squared(rho, grid) .* coefficient
    phihat = -rhohat
    zero_mode = k2 .== 0
    phihat[.!zero_mode] ./= k2[.!zero_mode]
    phihat[zero_mode] .= 0
    return ScalarField(real(ifft_spatial(phihat, grid)))
end

function electric_field_from_potential(phi::ScalarField, grid::Grid)
    e_components =
        [ScalarField(-differentiate(phi, grid, dir).data) for dir = 1:spatial_ndims(grid)]
    return vectorfield_from_spatial_components(e_components)
end

function fourier_filter(field::ScalarField, grid::Grid, cutoff_fraction::Real)
    fieldhat = fft_spatial(field.data, grid)
    k2 = spectral_wavenumber_squared(field, grid)
    kmax2 = maximum(k2) * cutoff_fraction^2
    fieldhat[k2 .> kmax2] .= 0
    return ScalarField(real(ifft_spatial(fieldhat, grid)))
end

# ── In-place Poisson solve using workspace ───────────────────────────────────

function _poisson_potential!(
    ws::PoissonSolverWorkspace,
    rho::ScalarField,
    coefficient::AbstractFloat,
)
    sw = ws.sw
    @. sw.fft_buf = rho.data
    for d = 1:length(sw.fwd_plans)
        sw.fwd_plans[d] * sw.fft_buf
    end
    @. sw.fft_buf = ifelse(ws.k2 == 0, complex(0.0), -sw.fft_buf / (coefficient * ws.k2))
    for d = 1:length(sw.inv_plans)
        sw.inv_plans[d] * sw.fft_buf
    end
    @. ws.phi_buf = real(sw.fft_buf)
    return ws.phi_buf
end

function _fourier_filter!(ws::PoissonSolverWorkspace, cutoff_fraction::Real)
    sw = ws.sw
    @. sw.fft_buf = ws.phi_buf
    for d = 1:length(sw.fwd_plans)
        sw.fwd_plans[d] * sw.fft_buf
    end
    kmax2 = maximum(ws.k2) * cutoff_fraction^2
    @. sw.fft_buf = ifelse(ws.k2 > kmax2, complex(0.0), sw.fft_buf)
    for d = 1:length(sw.inv_plans)
        sw.inv_plans[d] * sw.fft_buf
    end
    @. ws.phi_buf = real(sw.fft_buf)
    return ws.phi_buf
end

# ── solve_fields ─────────────────────────────────────────────────────────────

function solve_fields(moments::Moments, grid::Grid, solver::PoissonSolver)
    ws = _get_poisson_workspace(moments.rho, grid, solver.factor)
    _poisson_potential!(ws, moments.rho, solver.factor)
    _fourier_filter!(ws, 0.5)
    # E = -∇φ: differentiate phi_buf (wrapped as ScalarField) per direction
    phi_sf = ScalarField(ws.phi_buf)
    ndims_x = spatial_ndims(grid)
    for dir = 1:ndims_x
        _differentiate_impl!(ws.E_vf[dir].data, phi_sf, ws.sw, dir, true)
    end
    for dir = (ndims_x+1):3
        fill!(ws.E_vf[dir].data, 0.0)
    end
    return FieldSolution{typeof(ws.E_vf),typeof(phi_sf)}(
        ws.E_vf,
        ws.zero_vf3,
        ws.zero_vf3,
        phi_sf,
    )
end

function solve_fields(moments::Moments, grid::Grid, ::AdiabaticSolver)
    ws = _get_adiabatic_workspace(moments.rho, grid)
    ndims_x = spatial_ndims(grid)
    for dir = 1:ndims_x
        _differentiate_impl!(ws.E_vf[dir].data, moments.rho, ws.sw, dir, true)
    end
    return FieldSolution{typeof(ws.E_vf),typeof(ws.zero_phi)}(
        ws.E_vf,
        ws.zero_vf3,
        ws.zero_vf3,
        moments.rho,
    )
end
