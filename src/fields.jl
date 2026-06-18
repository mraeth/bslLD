struct ScalarField{DT,N,AT<:AbstractArray{DT,N}}
    data::AT

    function ScalarField(data::AbstractArray{DT,N}) where {DT,N}
        allocated = bslLD.backend_array(data)
        return new{eltype(allocated),ndims(allocated),typeof(allocated)}(allocated)
    end
end

import Base: +, -, *, transpose, adjoint, getindex, iterate, length, size


struct VectorField{DT,N,SF<:ScalarField{DT,N},NF}
    data::NTuple{NF,SF}

    function VectorField(data::Tuple{Vararg{<:ScalarField}})
        return VectorField(collect(data))
    end

    function VectorField(data::AbstractVector{<:ScalarField})
        isempty(data) && throw(ArgumentError("VectorField requires at least one component"))
        first_component = first(data)
        component_axes = axes(first_component.data)
        component_type = typeof(first_component)

        for component in data
            axes(component.data) == component_axes || throw(DimensionMismatch("..."))
            typeof(component) == component_type || throw(ArgumentError("..."))
        end

        tup = Tuple(data)  # convert Vector → NTuple at construction time
        return new{
            eltype(first_component.data),
            ndims(first_component.data),
            component_type,
            length(tup),
        }(
            tup,
        )
    end
end

function VectorField(data::AbstractVector{<:AbstractArray})
    return VectorField([ScalarField(bslLD.backend_array(x)) for x in data])
end


struct MatrixField{DT,N,SF<:ScalarField{DT,N},NR,NC,NF}
    data::NTuple{NF,SF}

    function MatrixField(data::AbstractMatrix{<:ScalarField})
        NR, NC = size(data)
        (NR >= 1 && NC >= 1) ||
            throw(ArgumentError("MatrixField requires at least one component"))
        first_component = first(data)
        component_axes = axes(first_component.data)
        component_type = typeof(first_component)
        for component in data
            axes(component.data) == component_axes ||
                throw(DimensionMismatch("MatrixField components must share axes"))
            typeof(component) == component_type ||
                throw(ArgumentError("MatrixField components must have the same type"))
        end
        tup = Tuple(data[i, j] for i = 1:NR for j = 1:NC)
        return new{
            eltype(first_component.data),
            ndims(first_component.data),
            component_type,
            NR,
            NC,
            NR*NC,
        }(
            tup,
        )
    end
end

function MatrixField(rows::AbstractVector{<:VectorField{DT,N,SF,NC}}) where {DT,N,SF,NC}
    isempty(rows) && throw(ArgumentError("MatrixField requires at least one row"))
    NR = length(rows)
    mat = Matrix{SF}(undef, NR, NC)
    for (i, row) in enumerate(rows)
        for j = 1:NC
            mat[i, j] = row[j]
        end
    end
    return MatrixField(mat)
end

function MatrixField(data::AbstractMatrix{<:AbstractArray})
    return MatrixField([
        ScalarField(bslLD.backend_array(data[i, j])) for
        i = 1:size(data, 1), j = 1:size(data, 2)
    ])
end

getindex(m::MatrixField{DT,N,SF,NR,NC,NF}, i::Int, j::Int) where {DT,N,SF,NR,NC,NF} =
    m.data[(i-1)*NC+j]
getindex(m::MatrixField{DT,N,SF,NR,NC,NF}, i::Int) where {DT,N,SF,NR,NC,NF} =
    VectorField(SF[m[i, j] for j = 1:NC])
size(::MatrixField{DT,N,SF,NR,NC,NF}) where {DT,N,SF,NR,NC,NF} = (NR, NC)
length(m::MatrixField) = length(m.data)
iterate(m::MatrixField, state...) = iterate(m.data, state...)

getindex(field::VectorField, i::Int) = field.data[i]
length(field::VectorField) = length(field.data)
iterate(field::VectorField, state...) = iterate(field.data, state...)

function +(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N}
    axes(a.data) == axes(b.data) ||
        throw(DimensionMismatch("ScalarField axes must match for addition"))
    return ScalarField(a.data .+ b.data)
end

function -(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N}
    axes(a.data) == axes(b.data) ||
        throw(DimensionMismatch("ScalarField axes must match for subtraction"))
    return ScalarField(a.data .- b.data)
end

function +(a::ScalarField{DT,N}, b::Number) where {DT,N}
    return ScalarField(a.data .+ b)
end

function +(a::Number, b::ScalarField{DT,N}) where {DT,N}
    return ScalarField(a .+ b.data)
end

function *(a::ScalarField{DT,N}, b::ScalarField{DT,N}) where {DT,N}
    axes(a.data) == axes(b.data) ||
        throw(DimensionMismatch("ScalarField axes must match for multiplication"))
    return ScalarField(a.data .* b.data)
end

function *(a::ScalarField{DT,N}, b::Number) where {DT,N}
    return ScalarField(a.data .* b)
end

function *(a::Number, b::ScalarField{DT,N}) where {DT,N}
    return ScalarField(a .* b.data)
end

function *(a::Number, b::VectorField{DT,N,SF,NF}) where {DT,N,SF<:ScalarField{DT,N},NF}
    return VectorField(map(component -> a * component, b.data))
end

function *(a::VectorField{DT,N,SF,NF}, b::Number) where {DT,N,SF<:ScalarField{DT,N},NF}
    return VectorField(map(component -> component * b, a.data))
end

function +(
    a::VectorField{DT,N,SF,NF},
    b::VectorField{DT,N,SF,NF},
) where {DT,N,SF<:ScalarField{DT,N},NF}
    return VectorField(map((ca, cb) -> ca + cb, a.data, b.data))
end

function -(
    a::VectorField{DT,N,SF,NF},
    b::VectorField{DT,N,SF,NF},
) where {DT,N,SF<:ScalarField{DT,N},NF}
    return VectorField(map((ca, cb) -> ca - cb, a.data, b.data))
end

function *(R::AbstractMatrix{<:Number}, v::VectorField{DT,N,SF,NF}) where {DT,N,SF,NF}
    NR_out = size(R, 1)
    size(R, 2) == NF || throw(
        DimensionMismatch("matrix columns $(size(R,2)) must match VectorField length $NF"),
    )
    result = SF[sum(R[i, j] * v[j] for j = 1:NF) for i = 1:NR_out]
    return VectorField(result)
end

function *(
    R::AbstractMatrix{<:Number},
    m::MatrixField{DT,N,SF,NR,NC,NF},
) where {DT,N,SF,NR,NC,NF}
    NR_out = size(R, 1)
    size(R, 2) == NR || throw(
        DimensionMismatch("matrix columns $(size(R,2)) must match MatrixField rows $NR"),
    )
    result = SF[sum(R[i, k] * m[k, j] for k = 1:NR) for i = 1:NR_out, j = 1:NC]
    return MatrixField(result)
end

function *(
    m::MatrixField{DT,N,SF,NR,NC,NF},
    R::AbstractMatrix{<:Number},
) where {DT,N,SF,NR,NC,NF}
    NC_out = size(R, 2)
    size(R, 1) == NC || throw(
        DimensionMismatch("matrix rows $(size(R,1)) must match MatrixField columns $NC"),
    )
    result = SF[sum(m[i, k] * R[k, j] for k = 1:NC) for i = 1:NR, j = 1:NC_out]
    return MatrixField(result)
end

transpose(m::MatrixField{DT,N,SF,NR,NC,NF}) where {DT,N,SF,NR,NC,NF} =
    MatrixField(SF[m[j, i] for i = 1:NC, j = 1:NR])

adjoint(m::MatrixField) = transpose(m)

function +(
    a::MatrixField{DT,N,SF,NR,NC,NF},
    b::MatrixField{DT,N,SF,NR,NC,NF},
) where {DT,N,SF,NR,NC,NF}
    return MatrixField(reshape(SF[a[i, j] + b[i, j] for i = 1:NR for j = 1:NC], NR, NC))
end

function -(
    a::MatrixField{DT,N,SF,NR,NC,NF},
    b::MatrixField{DT,N,SF,NR,NC,NF},
) where {DT,N,SF,NR,NC,NF}
    return MatrixField(reshape(SF[a[i, j] - b[i, j] for i = 1:NR for j = 1:NC], NR, NC))
end

function *(a::Number, b::MatrixField{DT,N,SF,NR,NC,NF}) where {DT,N,SF,NR,NC,NF}
    return MatrixField(reshape(SF[a * b[i, j] for i = 1:NR for j = 1:NC], NR, NC))
end

function *(a::MatrixField{DT,N,SF,NR,NC,NF}, b::Number) where {DT,N,SF,NR,NC,NF}
    return MatrixField(reshape(SF[a[i, j] * b for i = 1:NR for j = 1:NC], NR, NC))
end

function *(
    m::MatrixField{DT,N,SF,NR,NC,NF},
    v::VectorField{DT,N,SF,NC},
) where {DT,N,SF,NR,NC,NF}
    result = SF[sum(m[i, j] * v[j] for j = 1:NC) for i = 1:NR]
    return VectorField(result)
end

function empty_matrixfield(grid::Grid, NR::Int, NC::Int)
    dims = Tuple(length(ax) for ax in grid.xaxes)
    data = [ScalarField(zeros(Float64, dims)) for _ = 1:(NR*NC)]
    return MatrixField(reshape(data, NR, NC))
end

function empty_vectorfield(grid::Grid)
    dims = Tuple(length(axes) for axes in grid.xaxes)
    ncomp = length(grid.vaxes)
    data = [ScalarField(zeros(Float64, dims)) for _ = 1:ncomp]
    return VectorField(data)
end

function empty_scalarfield(grid::Grid)
    dims = Tuple(length(axes) for axes in grid.xaxes)
    data = bslLD.allocate(zeros(Float64, dims))
    return ScalarField(data)
end

function zero_vectorfield_like(vf::VectorField)
    return VectorField([ScalarField(zero.(c.data)) for c in vf])
end

function Base.copyto!(dst::VectorField, src::VectorField)
    for d = 1:ncomponents(dst)
        dst[d].data .= src[d].data
    end
    return dst
end

function Base.copyto!(dst::VectorField, src::AbstractVector{<:ScalarField})
    for d = 1:ncomponents(dst)
        dst[d].data .= src[d].data
    end
    return dst
end

function Base.materialize!(dst::VectorField, bc::Base.Broadcast.Broadcasted)
    copyto!(dst, Base.materialize(bc))
    return dst
end
