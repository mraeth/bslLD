
abstract type Distribution end

struct DistributionGrid{DT, PDT,NX,NV,NXNV,ID,AT<:AbstractArray{DT,NXNV}} <: Distribution
    data::AT
    m::PDT
    q::PDT
end

const DistributionGrid{DT,NX,NV,NXNV,ID,AT} =
    DistributionGrid{DT,Float64,NX,NV,NXNV,ID,AT}

const DistributionGrid1d1v{T,ID,AT} = DistributionGrid{T,1,1,2,ID,AT}
const DistributionGrid1d2v{T,ID,AT} = DistributionGrid{T,1,2,3,ID,AT}
const DistributionGrid2d2v{T,ID,AT} = DistributionGrid{T,2,2,4,ID,AT}



@inline thermal_velocity(f::DistributionGrid) = inv(sqrt(f.m))
@inline electric_acceleration_scale(f::DistributionGrid) = f.q / sqrt(f.m)

function _species_parameters(::Type{DT}, m::Real, q::Real) where {DT}
    m > 0 || throw(ArgumentError("distribution mass m must be positive"))
    return DT(m), DT(q)
end



function Distribution(
    grid::CartGrid,
    epsilon;
    m = 1.0,
    q = 1.0,
    initFuncx = (x -> (01.0 .+ epsilon * sin(2pi/(grid.xaxes[1][end]+grid.delta[1])*x))),
    initFuncv = (v->exp(-v^2 / 2) / sqrt(2*pi)),
    initFuncv1 = initFuncv,
)
    fct_sp(x) = initFuncx(x)
    fct_v(v) = initFuncv(v)
    dx = [fct_sp.(x) for x in grid.xaxes]
    dv = [fct_v.(x) for x in grid.vaxes]
    if length(dv)==2
        dv[2].=initFuncv1.(grid.vaxes[2])
    end
    da = vcat(dx, dv)
    data = bslLD.backend_array(outer_product(da))
    species_m, species_q = _species_parameters(Float64, m, q)
    return DistributionGrid{
        eltype(data),
        length(grid.xaxes),
        length(grid.vaxes),
        length(grid.xaxes) + length(grid.vaxes),
        Cart,
        typeof(data),
    }(
        data,
        species_m,
        species_q,
    )
end

function Distribution(
    grid::PolarGrid,
    epsilon;
    m = 1.0,
    q = 1.0,
    initFuncx = (x -> (01.0 .+ epsilon * sin(2pi/(grid.xaxes[1][end]+grid.delta[1])*x))),
    initFuncv = (v->exp(-v^2 / 2) / sqrt(2*pi)),
)
    fct_sp(x) = initFuncx(x)
    fct_v(v) = initFuncv(v)
    dx = [fct_sp.(x) for x in grid.xaxes]
    dv = [fct_v.(grid.vaxes[1]), sin.(grid.vaxes[2])]
    da = vcat(dx, dv)
    data = bslLD.backend_array(outer_product(da))
    species_m, species_q = _species_parameters(Float64, m, q)
    return DistributionGrid{
        Float64,
        length(grid.xaxes),
        length(grid.vaxes),
        length(grid.xaxes) + length(grid.vaxes),
        Polar,
        typeof(data),
    }(
        data,
        species_m,
        species_q,
    )
end

function compute_density(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    dim = ntuple(i -> NX + i, Val(NV))
    dv = prod(grid.delta[(1+NX):(NX+NV)])
    return ScalarField(
        reshape(sum(f.data, dims = dim) * dv, ntuple(i -> length(grid.xaxes[i]), Val(NX))),
    )
end

@inline function _moment_setup(
    f::DistributionGrid{DT,NX,NV,NXNV},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    vdims = ntuple(i -> NX + i, Val(NV))
    dv = prod(grid.delta[(NX+1):(NX+NV)])
    xsize = ntuple(i -> length(grid.xaxes[i]), Val(NX))
    return vdims, dv, xsize
end

function _current_arrays(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    vdims, dv, xsize = _moment_setup(f, grid)
    return ntuple(
        a -> begin
            v = collect(grid.vaxes[a])
            vshape = ntuple(k -> k == NX + a ? length(v) : 1, Val(NXNV))
            reshape(sum(f.data .* reshape(v, vshape), dims = vdims) * dv, xsize)
        end,
        Val(NV),
    )
end

function compute_current(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    return VectorField([collect(J) for J in _current_arrays(f, grid)])
end

function compute_current(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    phase::Real,
) where {DT,NX,NV,NXNV}
    1 <= NV <= 3 || error("This function expects 1 <= NV <= 3.")

    J = compute_current(f, grid)
    phi = -bslLD.electric_acceleration_scale(f) * phase
    Rot = R(grid.Bdir, phi)
    return Rot[1:NV, 1:NV] * J
end


function _momentum_tensor_arrays(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    vdims, dv, xsize = _moment_setup(f, grid)
    return [
        begin
            va = collect(grid.vaxes[a])
            vb = collect(grid.vaxes[b])
            vashape = ntuple(k -> k == NX + a ? length(va) : 1, Val(NXNV))
            vbshape = ntuple(k -> k == NX + b ? length(vb) : 1, Val(NXNV))
            reshape(
                sum(f.data .* reshape(va, vashape) .* reshape(vb, vbshape), dims = vdims) *
                dv,
                xsize,
            )
        end for a = 1:NV, b = 1:NV
    ]
end

function compute_momentum_tensor(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
) where {DT,NX,NV,NXNV}
    return MatrixField(_momentum_tensor_arrays(f, grid))
end

function compute_momentum_tensor(
    f::DistributionGrid{DT,NX,NV,NXNV,Cart},
    grid::CartGrid,
    phase::Real,
) where {DT,NX,NV,NXNV}
    1 <= NV <= 3 || error("This function expects 1 <= NV <= 3.")

    Pi = compute_momentum_tensor(f, grid)
    phi = -bslLD.electric_acceleration_scale(f) * phase
    Rot = R(grid.Bdir, phi)[1:NV, 1:NV]
    return Rot * Pi * Rot'
end


function compute_density(f::DistributionGrid, grid::PolarGrid)
    dim = Tuple(i for i = (length(grid.xaxes)+1):(length(grid.xaxes)+length(grid.vaxes)))
    dv = prod(grid.delta[(1+length(grid.xaxes)):end])
    return ScalarField(
        reshape(
            sum(f.data .* reshape(grid.vaxes[1], 1, :, 1), dims = dim)*dv,
            Tuple([length(axes) for axes in grid.xaxes]),
        ),
    )
end
