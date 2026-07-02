struct VacuumMaxwellParams{T}
    c::T
    ϵ0::T
    μ0::T
end

function VacuumMaxwellParams(; c::Real = 1.0, ϵ0::Real = 1.0, μ0::Real = 1.0)
    T = promote_type(typeof(c), typeof(ϵ0), typeof(μ0))
    return VacuumMaxwellParams{T}(T(c), T(ϵ0), T(μ0))
end

struct EMSolverVacuum{T<:AbstractFloat} <: AbstractFieldSolver
    params::VacuumMaxwellParams{T}
end
EMSolverVacuum(; c::Real = 1.0, ϵ0::Real = 1.0, μ0::Real = 1.0) =
    EMSolverVacuum(VacuumMaxwellParams(; c, ϵ0, μ0))

function maxwell_supported_layout(field::VectorField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)
    return (ndims_x, ncomp) in ((1, 2), (1, 3), (2, 3), (3, 3))
end

# ── EMVacuumWorkspace ────────────────────────────────────────────────────────

struct EMVacuumWorkspace{SW,EH,BH,CEH,CBH,KV,K2T,FT,CT}
    sw::SW
    Ehat::EH
    Bhat::BH
    curlEhat::CEH
    curlBhat::CBH
    kviews::KV       # reshaped k views for broadcasting in spectral_curl_hat
    k2::K2T          # Float64 field-sized — |k|²
    alpha2::FT       # Float64 field-sized — updated each call
    prefac_buf::FT
    diagonal_buf::FT
    scratch_c::CT    # single complex scratch for update step
end

function _make_em_vacuum_workspace(arr::AbstractArray, grid::Grid, ncomp::Int)
    ndims_x = spatial_ndims(grid)
    ndims_data = ndims(arr)
    sw = _get_spectral_workspace(arr, grid)

    Ehat = ntuple(_ -> similar(arr, Complex{Float64}), ncomp)
    Bhat = ntuple(_ -> similar(arr, Complex{Float64}), ncomp)
    curlEhat = ntuple(_ -> similar(arr, Complex{Float64}), ncomp)
    curlBhat = ntuple(_ -> similar(arr, Complex{Float64}), ncomp)

    kviews = ntuple(
        d -> reshape(sw.k[d], ntuple(i -> i == d ? length(sw.k[d]) : 1, ndims_data)),
        ndims_x,
    )
    k2 = fill!(similar(arr, Float64), 0.0)
    for d = 1:ndims_x
        k2 .+= kviews[d] .^ 2
    end

    alpha2 = fill!(similar(arr, Float64), 0.0)
    prefac_buf = fill!(similar(arr, Float64), 0.0)
    diagonal_buf = fill!(similar(arr, Float64), 0.0)
    scratch_c = similar(arr, Complex{Float64})

    return EMVacuumWorkspace(
        sw,
        Ehat,
        Bhat,
        curlEhat,
        curlBhat,
        kviews,
        k2,
        alpha2,
        prefac_buf,
        diagonal_buf,
        scratch_c,
    )
end

function _get_em_vacuum_workspace(sol::FieldSolution, grid::Grid)
    arr = sol.E[1].data
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(sol.E)
    key = (:em_vacuum, size(arr), eltype(arr), ndims_x, ncomp, Tuple(grid.delta[1:ndims_x]))
    lock(_solver_workspace_cache_lock) do
        get!(
            () -> _make_em_vacuum_workspace(arr, grid, ncomp),
            _solver_workspace_cache,
            key,
        )
    end
end

# ── Maxwell Crank–Nicolson step ──────────────────────────────────────────────

function _step_maxwell_cn!(
    E::VectorField,
    B::VectorField,
    grid::Grid,
    ws::EMVacuumWorkspace;
    dt::Real,
    params::VacuumMaxwellParams,
)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(E)
    c = params.c
    sw = ws.sw

    # Forward FFT all components into Ehat, Bhat
    for d = 1:ncomp
        @. sw.fft_buf = E[d].data
        for dim = 1:ndims_x
            sw.fwd_plans[dim] * sw.fft_buf
        end
        ws.Ehat[d] .= sw.fft_buf

        @. sw.fft_buf = B[d].data
        for dim = 1:ndims_x
            sw.fwd_plans[dim] * sw.fft_buf
        end
        ws.Bhat[d] .= sw.fft_buf
    end

    @. ws.alpha2 = (c * dt / 2)^2 * ws.k2
    @. ws.prefac_buf = 1.0 / (1.0 + ws.alpha2)
    @. ws.diagonal_buf = 1.0 - ws.alpha2

    _spectral_curl_hat!(ws.curlEhat, ws.Ehat, ws.kviews, ndims_x)
    _spectral_curl_hat!(ws.curlBhat, ws.Bhat, ws.kviews, ndims_x)

    for d = 1:ncomp
        @. ws.scratch_c =
            ws.prefac_buf * (ws.diagonal_buf * ws.Ehat[d] + c^2 * dt * ws.curlBhat[d])
        @. ws.Bhat[d] = ws.prefac_buf * (ws.diagonal_buf * ws.Bhat[d] - dt * ws.curlEhat[d])
        ws.Ehat[d] .= ws.scratch_c
    end

    # Inverse FFT back to real space
    for d = 1:ncomp
        @. sw.fft_buf = ws.Ehat[d]
        for dim = 1:ndims_x
            sw.inv_plans[dim] * sw.fft_buf
        end
        @. E[d].data = real(sw.fft_buf)

        @. sw.fft_buf = ws.Bhat[d]
        for dim = 1:ndims_x
            sw.inv_plans[dim] * sw.fft_buf
        end
        @. B[d].data = real(sw.fft_buf)
    end

    return nothing
end

function solve_fields!(
    sol::FieldSolution,
    ::Moments,
    grid::Grid,
    solver::EMSolverVacuum,
    dt::Real,
)
    ncomponents(sol.Enew) == ncomponents(sol.B) ||
        throw(ArgumentError("E and B must have the same number of components"))
    maxwell_supported_layout(sol.Enew, grid) ||
        throw(ArgumentError("unsupported E layout for Maxwell step"))
    maxwell_supported_layout(sol.B, grid) ||
        throw(ArgumentError("unsupported B layout for Maxwell step"))

    copyto!(sol.Enew, sol.E)
    ws = _get_em_vacuum_workspace(sol, grid)
    _step_maxwell_cn!(sol.Enew, sol.B, grid, ws; dt = dt, params = solver.params)
    return sol
end

function electromagnetic_energy(
    E::VectorField,
    B::VectorField;
    params::VacuumMaxwellParams = VacuumMaxwellParams(),
)
    ncomponents(E) == ncomponents(B) ||
        throw(ArgumentError("E and B must have the same number of components"))

    energy_e = zero(eltype(E[1].data))
    energy_b = zero(eltype(B[1].data))

    for d = 1:ncomponents(E)
        energy_e += sum(abs2, E[d].data)
        energy_b += sum(abs2, B[d].data)
    end

    return 0.5 * (params.ϵ0 * energy_e + energy_b / params.μ0)
end

function maxwell_constraints(E::VectorField, B::VectorField, grid::Grid)
    return div(E, grid), div(B, grid)
end
