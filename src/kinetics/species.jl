# A species is a mass m and charge q (ion units) plus a model: the model type carries the
# equations of motion and owns the whole state (a distribution, or the moments a fluid evolves).
# Methods dispatch on the model parameter, e.g. `push_free!(sp::Species{<:Any,<:Kinetic}, ...)`.
# Fields of the model are reachable from the species (`sp.dist`, `sp.delta_n`, ...).

abstract type AbstractModel end
# Models whose state is a distribution function on a phase-space grid.
abstract type DistributionModel <: AbstractModel end

struct Species{PDT<:AbstractFloat,M<:AbstractModel}
    m::PDT
    q::PDT
    model::M
end

# Full-orbit kinetic model (gyration in the rotating frame of the advectors). `xwall`: nothing (periodic x), or the
# Lagrange method of the x1 advection with `Mirror()` walls on the first and last x1 node (`Kinetic(dist; x_boundary =
# :mirror, order)`, used by the charge-form steps).
struct Kinetic{DG<:DistributionGrid,XW} <: DistributionModel
    dist::DG
    xwall::XW
end
function Kinetic(dist::DistributionGrid; x_boundary::Symbol = :periodic, order::Int = 8)
    x_boundary in (:periodic, :mirror) || throw(ArgumentError("x_boundary must be :periodic or :mirror"))
    return Kinetic(dist, x_boundary === :mirror ? Lagrange(order) : nothing)
end

function Species(m::Real, q::Real, model::AbstractModel)
    m > 0 || throw(ArgumentError("distribution mass m must be positive"))
    T = promote_type(typeof(float(m)), typeof(float(q)))
    return Species{T,typeof(model)}(T(m), T(q), model)
end
Species(m::Real, q::Real, dist::DistributionGrid) = Species(m, q, Kinetic(dist))
Species(model::AbstractModel; q::Real = 1, m::Real = 1) = Species(m, q, model)

const DistSpecies = Species{<:AbstractFloat,<:DistributionModel}
const KineticSpecies = Species{<:AbstractFloat,<:Kinetic}

function Base.getproperty(sp::Species, sym::Symbol)
    hasfield(Species, sym) && return getfield(sp, sym)
    model = getfield(sp, :model)
    sym === :data && return model.dist.data
    return getproperty(model, sym)
end
Base.propertynames(sp::Species) = (fieldnames(Species)..., propertynames(getfield(sp, :model))...)

@inline thermal_velocity(s::Species) = inv(sqrt(s.m))
@inline electric_acceleration_scale(s::Species) = s.q / sqrt(s.m)
gyro_frequency(s::Species, grid::Grid) = abs(s.q) / s.m * grid.b0

import PlasmaCore: backend_copy
backend_copy(model::Kinetic) = Kinetic(backend_copy(model.dist), model.xwall)
backend_copy(sp::Species) = Species(sp.m, sp.q, backend_copy(getfield(sp, :model)))
