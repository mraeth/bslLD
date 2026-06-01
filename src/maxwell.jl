struct VacuumMaxwellParams{T}
    c::T
    ϵ0::T
    μ0::T
end

function VacuumMaxwellParams(; c::Real=1.0, ϵ0::Real=1.0, μ0::Real=1.0)
    T = promote_type(typeof(c), typeof(ϵ0), typeof(μ0))
    return VacuumMaxwellParams{T}(T(c), T(ϵ0), T(μ0))
end

spatial_fft_dims(grid::Grid) = Tuple(1:spatial_ndims(grid))

function fft_spatial(data::AbstractArray, grid::Grid)
    return fft(data, spatial_fft_dims(grid))
end

function ifft_spatial(data::AbstractArray, grid::Grid)
    return ifft(data, spatial_fft_dims(grid))
end

function spectral_wavenumber_views(field::ScalarField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    ndims_data = ndims(field.data)
    return ntuple(ndims_x) do dir
        reshape(
            spectral_wavenumbers(field.data, grid, dir),
            ntuple(d -> d == dir ? size(field.data, dir) : 1, ndims_data),
        )
    end
end

function spectral_wave_number_squared(field::ScalarField, grid::Grid)
    kviews = spectral_wavenumber_views(field, grid)
    k2 = bslLD.backend_array(zeros(Float64, size(field.data)))
    for kview in kviews
        k2 .+= kview .^ 2
    end
    return k2
end

function spectral_curl_hat(fieldhat::Vector, kviews, ndims_x::Int)
    ncomp = length(fieldhat)

    if ndims_x == 1
        kx = kviews[1]
        if ncomp == 2
            return [
                -im .* kx .* fieldhat[2],
                im .* kx .* fieldhat[1],
            ]
        elseif ncomp == 3
            zero_component = zero.(fieldhat[1])
            return [
                zero_component,
                -im .* kx .* fieldhat[3],
                im .* kx .* fieldhat[2],
            ]
        end
    elseif ndims_x == 2
        kx, ky = kviews
        if ncomp == 3
            return [
                im .* ky .* fieldhat[3],
                -im .* kx .* fieldhat[3],
                im .* kx .* fieldhat[2] .- im .* ky .* fieldhat[1],
            ]
        end
    elseif ndims_x == 3
        kx, ky, kz = kviews
        if ncomp == 3
            return [
                im .* ky .* fieldhat[3] .- im .* kz .* fieldhat[2],
                im .* kz .* fieldhat[1] .- im .* kx .* fieldhat[3],
                im .* kx .* fieldhat[2] .- im .* ky .* fieldhat[1],
            ]
        end
    end

    throw(ArgumentError("unsupported Maxwell curl layout for $ndims_x spatial dimensions and $ncomp components"))
end

function maxwell_supported_layout(field::VectorField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)
    return (ndims_x, ncomp) in ((1, 2), (1, 3), (2, 3), (3, 3))
end

function step_maxwell_cn!(
    E::VectorField,
    B::VectorField,
    grid::Grid;
    dt::Real=grid.dt,
    params::VacuumMaxwellParams=VacuumMaxwellParams(),
)
    ncomponents(E) == ncomponents(B) || throw(ArgumentError("E and B must have the same number of components"))
    maxwell_supported_layout(E, grid) || throw(ArgumentError("unsupported E layout for Maxwell step"))
    maxwell_supported_layout(B, grid) || throw(ArgumentError("unsupported B layout for Maxwell step"))

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

    for d in 1:ncomp
        Ehat_new = prefac .* (diagonal .* Ehat[d] .+ c^2 * dt .* curlBhat[d])
        Bhat_new = prefac .* (diagonal .* Bhat[d] .- dt .* curlEhat[d])

        E[d].data .= real(ifft_spatial(Ehat_new, grid))
        B[d].data .= real(ifft_spatial(Bhat_new, grid))
    end

    return nothing
end

function electromagnetic_energy(E::VectorField, B::VectorField; params::VacuumMaxwellParams=VacuumMaxwellParams())
    ncomponents(E) == ncomponents(B) || throw(ArgumentError("E and B must have the same number of components"))

    energy_e = zero(eltype(E[1].data))
    energy_b = zero(eltype(B[1].data))

    for d in 1:ncomponents(E)
        energy_e += sum(abs2, E[d].data)
        energy_b += sum(abs2, B[d].data)
    end

    return 0.5 * (params.ϵ0 * energy_e + energy_b / params.μ0)
end

function maxwell_constraints(E::VectorField, B::VectorField, grid::Grid)
    return div(E, grid), div(B, grid)
end

function _step_semi_implicit_em!(
    E::VectorField,
    B::VectorField,
    J_i_perp::VectorField,
    Pi_diff_z::VectorField,
    grid::Grid,
    solver,
    dt::Real,
)
    ndims_x  = spatial_ndims(grid)
    perp_dirs = [d for d in 1:ndims_x if d != grid.Bdir]

    # Step 1: Explicit Faraday — B^{n+1} = B^n - dt * curl(E^n)
    curlE = curl(E, grid)
    for d in 1:ncomponents(B)
        B[d].data .-= dt .* curlE[d].data
    end

    # Step 2: Implicit E_⊥ update — pointwise 2×2 solve per grid point
    # [I - (dt/μ)R] E_⊥^{n+1} = RHS_⊥
    # where R(a₁,a₂) = a × ê_z = (a₂, -a₁)
    curlB = curl(B, grid)   # uses B^{n+1}
    α  = dt / solver.mu
    c  = 2 / solver.beta_i  # coefficient for (∇×B)_⊥ in RHS

    if length(perp_dirs) >= 2
        d1, d2 = perp_dirs[1], perp_dirs[2]
        rhs1 = E[d1].data .+ α .* (c .* curlB[d1].data .- J_i_perp[d1].data)
        rhs2 = E[d2].data .+ α .* (c .* curlB[d2].data .- J_i_perp[d2].data)
        denom = 1 + α^2
        E[d1].data .= (rhs1 .+ α .* rhs2) ./ denom
        E[d2].data .= (rhs2 .- α .* rhs1) ./ denom
    elseif length(perp_dirs) == 1
        d1 = perp_dirs[1]
        rhs1 = E[d1].data .+ α .* (c .* curlB[d1].data .- J_i_perp[d1].data)
        E[d1].data .= rhs1   # R degenerates in 1D: no cross-coupling
    end

    # Step 3: Helmholtz solve for E_z
    # [∇_⊥² - β_i/2*(1+1/μ)] E_z^{n+1} = β_i/2 * ∇·Pi_diff_z + ∂_z(∇_⊥·E_⊥^{n+1})
    div_Pi = div(Pi_diff_z, grid)

    div_E_perp = differentiate(E[perp_dirs[1]], grid, perp_dirs[1])
    for d in perp_dirs[2:end]
        div_E_perp = div_E_perp + differentiate(E[d], grid, d)
    end

    if ndims_x >= 3
        dz_div_E_perp = differentiate(div_E_perp, grid, grid.Bdir)
    else
        dz_div_E_perp = ScalarField(zero.(div_E_perp.data))
    end

    rhs_data = (solver.beta_i / 2) .* div_Pi.data .+ dz_div_E_perp.data
    rhs_hat  = fft_spatial(rhs_data, grid)

    kviews  = spectral_wavenumber_views(E[1], grid)
    kperp2  = kviews[perp_dirs[1]] .^ 2
    for d in perp_dirs[2:end]
        kperp2 = kperp2 .+ kviews[d] .^ 2
    end

    helmholtz_op = .-kperp2 .- (solver.beta_i / 2) * (1 + 1 / solver.mu)
    E[grid.Bdir].data .= real(ifft_spatial(rhs_hat ./ helmholtz_op, grid))

    return nothing
end
