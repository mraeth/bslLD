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
