struct ScalarField{DT,N}
    data :: Array{DT,N}

    function ScalarField(data::Array{DT,N}) where {DT,N}
        return new{DT,N}(data)
    end
end

import Base: +, *

struct VectorField{DT,N}
    data :: Vector{Array{DT,N}}

    function VectorField(data::Vector{Array{DT,N}}) where {DT,N}
        return new{DT,N}(data)
    end
end

function +(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N}
    axes(a.data) == axes(b.data) || throw(DimensionMismatch("ScalarField axes must match for addition"))
    return ScalarField(a.data .+ b.data)
end

function +(a::ScalarField{DT,N}, b::Number) where {DT,N}
    return ScalarField(a.data .+ b)
end

function +(a::Number, b::ScalarField{DT,N}) where {DT,N}
    return ScalarField(a .+ b.data)
end

function *(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N}
    axes(a.data) == axes(b.data) || throw(DimensionMismatch("ScalarField axes must match for multiplication"))
    return ScalarField(a.data .* b.data)
end

function *(a::ScalarField{DT,N}, b::Number) where {DT,N}
    return ScalarField(a.data .* b)
end

function *(a::Number, b::ScalarField{DT,N}) where {DT,N}
    return ScalarField(a .* b.data)
end

function empty_vectorfield(grid::Grid)
    dims = Tuple( length(axes) for axes in grid.xaxes )
    ncomp = length(grid.vaxes)
    data = [ zeros(Float64, dims) for i in 1:ncomp ]
    return VectorField(data)
end

function empty_scalarfield(grid::Grid)
    dims = Tuple( length(axes) for axes in grid.xaxes )
    data = zeros(Float64, dims)
    return ScalarField(data)
end

function poisson(rho::ScalarField{T,N}, grid::Grid) where {T,N}
    n = size(rho.data, 1)
    k = -2pi/grid.max[1]*fftfreq(length(rho.data),length(rho.data))
    k2 = k.^2
    k2[1] = 1   
    phi_data = -real( ifft( fft(rho.data) ./ k2 ) )

    return ScalarField(phi_data)
end


function adiabatic(rho::ScalarField{T,N}, grid::Grid) where {T,N}
    return ScalarField(copy(rho.data))
end


function compute_e(rho::ScalarField{T,N}, grid::Grid) where {T,N}
    n = size(rho.data, 1)
    k = -2pi/grid.max[1]*fftfreq(length(rho.data),length(rho.data))


    # Ex = -∂φ/∂x = -ik * φ̂ = -k * Im[FFT]
    Ex = real( ifft( -im .* k .* fft(rho.data) ) )
    Ey = 0*Ex
    return VectorField([Ex,Ey])
end
