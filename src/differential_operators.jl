struct DifferentiateContext{KT, ST}
    k::KT
    sizes::ST
    dir::Int
end

Adapt.@adapt_structure DifferentiateContext

@inline function (ctx::DifferentiateContext)(index::Int)
    return compute_derivative_multiplier(ctx, index)
end

@inline function compute_derivative_multiplier(ctx::DifferentiateContext, index::Int)
    idxs = index_1d_to_nd(index, ctx.sizes)
    return im * ctx.k[idxs[ctx.dir]]
end

@kernel function differentiate_kernel!(fhat, ctx)
    i = @index(Global)

    if i <= length(fhat)
        @inbounds fhat[i] *= ctx(i)
    end
end

function spectral_wavenumbers(field_data::AbstractArray, grid::Grid, dir::Int)
    1 <= dir <= length(grid.xaxes) || throw(ArgumentError("direction $dir is outside the spatial grid dimensions"))
    n = size(field_data, dir)
    axis_length = n * grid.delta[dir]
    k = similar(field_data, Float64, n)
    copyto!(k, collect((2pi / axis_length) .* fftfreq(n, n)))
    return k
end

function differentiate(field::ScalarField{T, N}, grid::Grid, dir::Int) where {T, N}
    1 <= N <= 3 || throw(ArgumentError("differentiate currently supports 1D, 2D, and 3D ScalarFields"))
    1 <= dir <= N || throw(ArgumentError("direction $dir is outside the field dimensions"))
    dir <= length(grid.xaxes) || throw(ArgumentError("direction $dir is outside the spatial grid dimensions"))
    return _differentiate_impl(field, grid, dir, bslLD.backend())
end

function _differentiate_impl(field::ScalarField{T, N}, grid::Grid, dir::Int, exec) where {T, N}
    kernel! = differentiate_kernel!(exec)
    fhat = fft(field.data, dir)
    k = spectral_wavenumbers(field.data, grid, dir)
    ctx = DifferentiateContext(k, size(fhat), dir)
    kernel!(fhat, ctx; ndrange=length(fhat))
    KernelAbstractions.synchronize(exec)
    return ScalarField(real(ifft(fhat, dir)))
end

spatial_ndims(grid::Grid) = length(grid.xaxes)
ncomponents(field::VectorField) = length(field)

function grad(field::ScalarField{T, N}, grid::Grid) where {T, N}
    ndirs = spatial_ndims(grid)
    ndirs >= 1 || throw(ArgumentError("grad requires at least one spatial dimension"))
    return VectorField([differentiate(field, grid, dir) for dir in 1:ndirs])
end

function div(field::VectorField{T, N}, grid::Grid) where {T, N}
    ndirs = spatial_ndims(grid)
    ncomp = ncomponents(field)
    ndirs >= 1 || throw(ArgumentError("div requires at least one spatial dimension"))
    ncomp >= ndirs || throw(ArgumentError("div on a $ndirs-D grid requires at least $ndirs vector components"))

    result = differentiate(field[1], grid, 1)
    for dir in 2:ndirs
        result = result + differentiate(field[dir], grid, dir)
    end

    return result
end

function div(field::MatrixField{DT,N,SF,NR,NC,NF}, grid::Grid) where {DT,N,SF,NR,NC,NF}
    ndirs = spatial_ndims(grid)
    ndirs >= 1 || throw(ArgumentError("div requires at least one spatial dimension"))
    NC >= ndirs || throw(ArgumentError("div on a $ndirs-D grid requires at least $ndirs matrix columns"))

    rows = Vector{SF}(undef, NR)
    for i in 1:NR
        row_result = differentiate(field[i, 1], grid, 1)
        for dir in 2:ndirs
            row_result = row_result + differentiate(field[i, dir], grid, dir)
        end
        rows[i] = row_result
    end

    return VectorField(rows)
end

function curl(field::VectorField, grid::Grid)
    ndims_x = spatial_ndims(grid)
    ncomp = ncomponents(field)

    if ndims_x == 1
        ncomp in (2, 3) || throw(ArgumentError("1D curl requires 2 or 3 vector components"))

        if ncomp == 2
            df1_dx = differentiate(field[1], grid, 1)
            df2_dx = differentiate(field[2], grid, 1)
            return VectorField([
                ScalarField(-df2_dx.data),
                ScalarField(df1_dx.data),
            ])
        end

        df2_dx = differentiate(field[2], grid, 1)
        df3_dx = differentiate(field[3], grid, 1)
        zero_component = ScalarField(zero.(df2_dx.data))
        return VectorField([
            zero_component,
            ScalarField(-df3_dx.data),
            ScalarField(df2_dx.data),
        ])
    elseif ndims_x == 2
        ncomp in (2, 3) || throw(ArgumentError("2D curl requires 2 or 3 vector components"))

        dv_dx = differentiate(field[2], grid, 1)
        du_dy = differentiate(field[1], grid, 2)

        if ncomp == 2
            return ScalarField(dv_dx.data .- du_dy.data)
        end

        dw_dx = differentiate(field[3], grid, 1)
        dw_dy = differentiate(field[3], grid, 2)
        return VectorField([
            ScalarField(dw_dy.data),
            ScalarField(-dw_dx.data),
            ScalarField(dv_dx.data .- du_dy.data),
        ])
    elseif ndims_x == 3
        ncomp == 3 || throw(ArgumentError("3D curl requires exactly 3 vector components"))

        dw_dy = differentiate(field[3], grid, 2)
        dv_dz = differentiate(field[2], grid, 3)
        du_dz = differentiate(field[1], grid, 3)
        dw_dx = differentiate(field[3], grid, 1)
        dv_dx = differentiate(field[2], grid, 1)
        du_dy = differentiate(field[1], grid, 2)

        return VectorField([
            ScalarField(dw_dy.data .- dv_dz.data),
            ScalarField(du_dz.data .- dw_dx.data),
            ScalarField(dv_dx.data .- du_dy.data),
        ])
    end

    throw(ArgumentError("curl currently supports 1D, 2D, and 3D grids"))
end
