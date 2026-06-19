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

function _step_maxwell_cn!(
    E::VectorField,
    B::VectorField,
    grid::Grid;
    dt::Real = grid.dt,
    params::VacuumMaxwellParams = VacuumMaxwellParams(),
)
    ncomponents(E) == ncomponents(B) ||
        throw(ArgumentError("E and B must have the same number of components"))
    maxwell_supported_layout(E, grid) ||
        throw(ArgumentError("unsupported E layout for Maxwell step"))
    maxwell_supported_layout(B, grid) ||
        throw(ArgumentError("unsupported B layout for Maxwell step"))

    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(E)
    c = params.c

    Ehat = [fft_spatial(component.data, grid) for component in E]
    Bhat = [fft_spatial(component.data, grid) for component in B]

    kviews = spectral_wavenumber_views(E[1], grid)
    k2 = spectral_wave_number_squared(E[1], grid)
    alpha2 = (c * dt / 2)^2 .* k2
    prefac = 1.0 ./ (1.0 .+ alpha2)
    diagonal = 1.0 .- alpha2

    curlEhat = spectral_curl_hat(Ehat, kviews, ndims_x)
    curlBhat = spectral_curl_hat(Bhat, kviews, ndims_x)

    for d = 1:ncomp
        Ehat_new = prefac .* (diagonal .* Ehat[d] .+ c^2 * dt .* curlBhat[d])
        Bhat_new = prefac .* (diagonal .* Bhat[d] .- dt .* curlEhat[d])

        E[d].data .= real(ifft_spatial(Ehat_new, grid))
        B[d].data .= real(ifft_spatial(Bhat_new, grid))
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
    copyto!(sol.Enew, sol.E)
    _step_maxwell_cn!(sol.Enew, sol.B, grid; dt = dt, params = solver.params)
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
