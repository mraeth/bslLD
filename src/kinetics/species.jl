struct Species{PDT<:AbstractFloat, DG<:DistributionGrid}
    m::PDT
    q::PDT
    dist::DG
end

function Species(m::Real, q::Real, dist::DistributionGrid)
    m > 0 || throw(ArgumentError("distribution mass m must be positive"))
    T = promote_type(typeof(float(m)), typeof(float(q)))
    return Species{T, typeof(dist)}(T(m), T(q), dist)
end

function Base.getproperty(sp::Species, sym::Symbol)
    sym === :data && return getfield(sp, :dist).data
    return getfield(sp, sym)
end

@inline thermal_velocity(s::Species)            = inv(sqrt(s.m))
@inline electric_acceleration_scale(s::Species) = s.q / sqrt(s.m)
gyro_frequency(s::Species, grid::Grid)          = abs(s.q) / s.m * grid.b0
