struct Grid{T, XT, VT, MT, TT, ITT, IXT, CTT, ID}
    dt::T
    xaxes::XT
    vaxes::VT
    max::MT
    min::MT
    delta::MT
    time::TT
    b0::T
    itime::ITT
    index::IXT
    curr_time::CTT
    Bdir::Int
end

abstract type Cart end
abstract type Polar end

const CartGrid = Grid{T, XT, VT, MT, TT, ITT, IXT, CTT, Cart} where {T, XT, VT, MT, TT, ITT, IXT, CTT}
const PolarGrid = Grid{T, XT, VT, MT, TT, ITT, IXT, CTT, Polar} where {T, XT, VT, MT, TT, ITT, IXT, CTT}

function Adapt.adapt_structure(to, grid::Grid{T, XT, VT, MT, TT, ITT, IXT, CTT, ID}) where {T, XT, VT, MT, TT, ITT, IXT, CTT, ID}
    dt = Adapt.adapt(to, grid.dt)
    xaxes = Adapt.adapt(to, grid.xaxes)
    vaxes = Adapt.adapt(to, grid.vaxes)
    maxv = Adapt.adapt(to, grid.max)
    minv = Adapt.adapt(to, grid.min)
    delta = Adapt.adapt(to, grid.delta)
    time = Adapt.adapt(to, grid.time)
    b0 = Adapt.adapt(to, grid.b0)
    itime = Adapt.adapt(to, grid.itime)
    index = Adapt.adapt(to, grid.index)
    curr_time = Adapt.adapt(to, grid.curr_time)
    return Grid{typeof(dt), typeof(xaxes), typeof(vaxes), typeof(maxv), typeof(time), typeof(itime), typeof(index), typeof(curr_time), ID}(
        dt,
        xaxes,
        vaxes,
        maxv,
        minv,
        delta,
        time,
        b0,
        itime,
        index,
        curr_time,
        grid.Bdir,
    )
end

_axis_tuple(axes::Vector) = Tuple(axes)
_meta_vector(values::AbstractVector{T}) where {T} = SVector{length(values), T}(values)

function set_time_index(grid::Grid{T, XT, VT, MT, TT, ITT, IXT, CTT, ID}, i::Integer) where {T, XT, VT, MT, TT, ITT, IXT, CTT, ID}
    return Grid{T, XT, VT, MT, TT, ITT, typeof(SVector(Int(i))), CTT, ID}(
        grid.dt,
        grid.xaxes,
        grid.vaxes,
        grid.max,
        grid.min,
        grid.delta,
        grid.time,
        grid.b0,
        grid.itime,
        SVector(Int(i)),
        grid.curr_time,
        grid.Bdir,
    )
end

function Grid(
    etaMin::Vector{Float64},
    etaMax::Vector{Float64},
    N::Vector{Int64},
    dt::Float64,
    nt::Int64,
    nx::Int64,
    b0=1.0::Float64,
    Bdir=3::Int;
    type=Cart,
)
    if type == Cart
        delta = (etaMax .- etaMin) ./ N
        xaxes = [range(etaMin[i], step=delta[i], length=N[i]) for i in 1:nx]
        vaxes = [range(etaMin[i], step=delta[i], length=N[i] + 1) for i in (nx + 1):length(etaMin)]

        return Grid{
            Float64,
            typeof(_axis_tuple(xaxes)),
            typeof(_axis_tuple(vaxes)),
            typeof(_meta_vector(etaMax)),
            typeof(range(0.0, step=dt, length=nt + 1)),
            UnitRange{Int},
            SVector{1, Int},
            SVector{2, Float64},
            Cart,
        }(
            dt,
            _axis_tuple(xaxes),
            _axis_tuple(vaxes),
            _meta_vector(etaMax),
            _meta_vector(etaMin),
            _meta_vector(delta),
            range(0.0, step=dt, length=nt + 1),
            b0,
            1:nt,
            SVector(1),
            SVector(0.0, 0.0),
            Bdir,
        )
    else
        vpmax = etaMax[nx + 1]
        deltavp = vpmax / N[nx + 1]
        deltaphi = 2pi / N[nx + 2]
        delta = vcat((etaMax .- etaMin) ./ N[1:nx], [deltavp, deltaphi])
        xaxes = [range(etaMin[i], step=delta[i], length=N[i]) for i in 1:nx]
        vaxes = (
            range(deltavp, step=deltavp, length=N[nx + 1]),
            range(0.0, step=deltaphi, length=N[nx + 2]),
        )

        return Grid{
            Float64,
            typeof(_axis_tuple(xaxes)),
            typeof(vaxes),
            typeof(_meta_vector(etaMax)),
            typeof(range(0.0, step=dt, length=nt + 1)),
            UnitRange{Int},
            SVector{1, Int},
            SVector{2, Float64},
            Polar,
        }(
            dt,
            _axis_tuple(xaxes),
            vaxes,
            _meta_vector(etaMax),
            _meta_vector(etaMin),
            _meta_vector(delta),
            range(0.0, step=dt, length=nt + 1),
            b0,
            1:nt,
            SVector(1),
            SVector(0.0, 0.0),
            Bdir,
        )
    end
end

outer_product(vs) = .*([reshape(vs[d], (ntuple(Returns(1), d - 1)..., :)) for d in 1:length(vs)]...)
