struct ScalarField{DT,N,AT<:AbstractArray{DT,N}}
    data :: AT

    function ScalarField(data::AbstractArray{DT,N}) where {DT,N}
        allocated = bslLD.backend_array(data)
        return new{eltype(allocated), ndims(allocated), typeof(allocated)}(allocated)
    end
end

import Base: +, *, getindex, iterate, length


struct VectorField{DT, N, SF<:ScalarField{DT,N}, NF}
    data :: NTuple{NF, SF}

    function VectorField(data::AbstractVector{<:ScalarField})
        isempty(data) && throw(ArgumentError("VectorField requires at least one component"))
        first_component = first(data)
        component_axes = axes(first_component.data)
        component_type = typeof(first_component)

        for component in data
            axes(component.data) == component_axes || throw(DimensionMismatch("..."))
            typeof(component) == component_type    || throw(ArgumentError("..."))
        end

        tup = Tuple(data)  # convert Vector → NTuple at construction time
        return new{eltype(first_component.data), ndims(first_component.data),
                   component_type, length(tup)}(tup)
    end
end

function VectorField(data::AbstractVector{<:AbstractArray})
    return VectorField(ScalarField.(data))
end

getindex(field::VectorField, i::Int) = field.data[i]
length(field::VectorField) = length(field.data)
iterate(field::VectorField, state...) = iterate(field.data, state...)

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
    dims = Tuple(length(axes) for axes in grid.xaxes)
    ncomp = length(grid.vaxes)
    data = [ScalarField(zeros(Float64, dims)) for _ in 1:ncomp]
    return VectorField(data)
end

function empty_scalarfield(grid::Grid)
    dims = Tuple(length(axes) for axes in grid.xaxes)
    data = bslLD.allocate(zeros(Float64, dims))
    return ScalarField(data)
end
