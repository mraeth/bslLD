function displace_polar_velocity!(
    vrp::Base.RefValue{<:Real},
    phip::Base.RefValue{<:Real},
    vr::Real,
    phi::Real,
    delta_vx::Real,
    delta_vy::Real
)
    vx = vr * cos(phi)
    vy = vr * sin(phi)

    vx_prime = vx + delta_vx
    vy_prime = vy + delta_vy

    vrp[] = sqrt(vx_prime^2 + vy_prime^2)
    phip[] = atan(vy_prime, vx_prime)

    return nothing
end