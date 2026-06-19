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
            return [-im .* kx .* fieldhat[2], im .* kx .* fieldhat[1]]
        elseif ncomp == 3
            zero_component = zero.(fieldhat[1])
            return [zero_component, -im .* kx .* fieldhat[3], im .* kx .* fieldhat[2]]
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

    throw(
        ArgumentError(
            "unsupported spectral curl layout for $ndims_x spatial dimensions and $ncomp components",
        ),
    )
end
