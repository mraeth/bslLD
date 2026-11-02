# Building f from a source over a ramp (bsl6d's rampTime/rampSTD) and setting initial conditions.

"""
    deposit!(f, grid, target, t; weight, dt)

Add `weight * dt * target(grid, t)` to the distribution function `f` (a `Species` or a
distribution). `target(grid, t)` returns an array of the size of `f.data`.

In the time loop, call it with a time-dependent weight to build `f` from a source, e.g.
`deposit!(f, grid, target, t; weight = ramp(t), dt = dt/2)`. Once before the loop it sets the
initial condition: `deposit!(f, grid, target, 0.0; weight = 1, dt = 1)` on an `f` that starts at
zero. `target` is not evaluated if `weight * dt` is zero.
"""
function deposit!(f::bslLD.DistributionGrid, grid::bslLD.Grid, target, t; weight, dt)
    w = weight * dt
    iszero(w) && return f
    add = target(grid, t)
    size(add) == size(f.data) || throw(
        DimensionMismatch(
            "target(grid, t) has size $(size(add)), the distribution function has $(size(f.data))",
        ),
    )
    f.data .+= w .* add
    return f
end

function deposit!(sp::bslLD.Species, args...; kwargs...)
    deposit!(sp.dist, args...; kwargs...)
    return sp
end

"""
    gaussian_ramp(t_mid, σ) -> (t -> weight)

Gaussian weight centred at `t_mid` with standard deviation `σ`, restricted to the window
`0 ≤ t < 2 t_mid` and normalised so that its integral over the window is 1 (bsl6d's
`rampTime/2`, `rampSTD`). Zero outside the window.
"""
function gaussian_ramp(t_mid, σ)
    t_mid > 0 && σ > 0 || throw(ArgumentError("t_mid and σ must be positive"))
    norm = erf(t_mid / (σ * sqrt(2)))          # ∫ over the window of the unit Gaussian pdf
    return t ->
        (0 <= t < 2 * t_mid) ? exp(-(t - t_mid)^2 / (2 * σ^2)) / (σ * sqrt(2pi) * norm) :
        zero(float(t))
end
