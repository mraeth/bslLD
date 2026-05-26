mutable struct SimulationTime{T<:AbstractFloat}
    dt::T
    current_T::T
    final_T::T
    step::Int
    phase::T
    nmax::Int
    gyro_frequency::T
    wall_start_ns::UInt64
end

function SimulationTime(
    dt::T,
    final_T::T;
    nmax::Integer = typemax(Int),
    gyro_frequency = zero(T),
    current_T = zero(T),
    step::Integer = 0,
    phase = zero(T),
) where {T<:AbstractFloat}

    return SimulationTime{T}(
        dt,
        T(current_T),
        final_T,
        Int(step),
        T(phase),
        Int(nmax),
        T(gyro_frequency),
        time_ns(),
    )
end

function SimulationTime(dt::Real, final_T::Real; kwargs...)
    T = promote_type(typeof(float(dt)), typeof(float(final_T)))
    return SimulationTime(T(dt), T(final_T); kwargs...)
end

function advance!(t::SimulationTime{T}; wrap_phase::Bool = true) where {T<:AbstractFloat}
    t.current_T += t.dt
    t.step += 1
    t.phase += t.gyro_frequency * t.dt

    if wrap_phase
        t.phase = mod(t.phase, T(2π))
    end

    return t
end

function continue_advection(t::SimulationTime)
    return t.current_T < t.final_T && t.step < t.nmax
end

function elapsed_seconds(t::SimulationTime)
    return Float64(time_ns() - t.wall_start_ns) * 1.0e-9
end

function reset_timer!(t::SimulationTime)
    t.wall_start_ns = time_ns()
    return t
end