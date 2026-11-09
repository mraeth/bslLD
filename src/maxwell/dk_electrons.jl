# Linear drift-kinetic model, the first step towards a full DK species model.
#
# State δF(x, v∥), perturbation of B*∥ f around n0 = 1, on its own 1V grid (the x axes of the
# field grid and one velocity axis along B, in units of 1/√m). Maxwellian F0 with temperature
# T_e, no perpendicular temperature (no mirror force), uniform B0 = b0 b̂. Linearised
# guiding-centre Liouville equation, δF = δf + F0 δB∥/b0:
#
#   ∂t δF + v ∂∥ δf = -F0 b̂·∇×E / b0 - (q/√m) E∥ F0'
#
# The streaming acts on δf, not on δF (streaming δF would add the isotropic mirror force
# -v ∂∥(F0 δB∥)/b0), so `push_free!` needs B∥: δF ← S(δF − F0 B∥/b0) + F0 B∥/b0, with B = B^n in
# the first half step and B^{n+1} in the second. In terms of δf the curl term is the Jacobian.
# Streaming is the kinetic parallel advection (`advectX!` along Bdir with `vdirs = (Bdir,)`);
# the kick is the linear source `add_linear_dk_kick!` (sources.jl), replaced by `advectV!` and
# E×B advection in the full model. F0, F0' are normalised on the grid, Σ F0 Δv = 1 and
# Σ v F0' Δv = -1, so that k∥ = 0 and T_e → 0 reproduce the cold drift fluid to round-off.
# A drift species: E×B current and polarisation density as for `ColdFluid{DriftFluid}`.

struct LinearDK{DG<:DistributionGrid,G<:Grid,MT} <: DistributionModel
    dist::DG      # δF on `grid`
    grid::G       # x axes of the field grid, one v∥ axis
    T_e::Float64
    c0::Float64   # normalisation of F0
    c1::Float64   # normalisation of F0'
    method::MT    # Fourier() (exact stage response) or Lagrange(order) for the streaming
end

"""
    LinearDK(field_grid; T_e, Nv = 64, vmax = 6, method = Fourier())

Linear drift-kinetic model on the x axes of `field_grid` with a v∥ axis of `Nv` intervals
(`Nv + 1` nodes) on ±`vmax` √T_e, δF = 0. Species: `Species(LinearDK(grid; T_e); q = -1, m = μ)`.
"""
function LinearDK(field_grid::CartGrid; T_e::Real, Nv::Int = 64, vmax::Real = 6, method = Fourier())
    T_e > 0 || throw(ArgumentError("T_e must be positive"))
    nx = length(field_grid.xaxes)
    vm = vmax * sqrt(T_e)
    grid = Grid(vcat(Vector{Float64}(field_grid.min[1:nx]), -vm),
                vcat(Vector{Float64}(field_grid.max[1:nx]), vm),
                vcat(collect(length.(field_grid.xaxes)), Nv), nx, field_grid.b0, field_grid.Bdir)
    sf = ScalarField(backend_array(zeros(Float64, length.(grid.xaxes)..., Nv + 1)))
    dist = DistributionGridImpl{nx,1,Cart,typeof(sf)}(sf)
    v, dv = collect(grid.vaxes[1]), grid.delta[nx+1]
    g = @. exp(-v^2 / (2T_e)) / sqrt(2π * T_e)
    c0 = 1 / (dv * sum(g))
    c1 = 1 / (dv * sum(@. v^2 / T_e * g))
    return LinearDK(dist, grid, Float64(T_e), c0, c1, method)
end

const LinearDKSpecies = Species{<:AbstractFloat,<:LinearDK}

# Normalised F0 and F0' on the v∥ nodes (the formulas of the kick kernel).
function _dk_maxwellian(md::LinearDK)
    v, T = collect(md.grid.vaxes[1]), md.T_e
    g = @. exp(-v^2 / (2T)) / sqrt(2π * T)
    return md.c0 .* g, @.(-md.c1 * v / T * g)
end

_dk_dv(md::LinearDK) = md.grid.delta[length(md.grid.xaxes)+1]

function push_free!(sp::LinearDKSpecies, grid::Grid, dt::Real, half::Int; phase = 0, B = nothing)
    B === nothing && throw(ArgumentError("LinearDK streaming needs B (the Jacobian part F0 δB∥ does not stream)"))
    md = getfield(sp, :model)
    pz = md.grid.Bdir
    pz <= length(md.grid.xaxes) || return sp      # no parallel axis: k∥ = 0
    nx = length(md.grid.xaxes)
    F0 = reshape(_dk_maxwellian(md)[1], ntuple(_ -> 1, nx)..., :)
    jac = (B[pz].data ./ grid.b0) .* F0
    md.dist.data .-= jac
    t = SimulationTime(dt, dt)
    t.fraction_dt = 0.5
    advectX!(sp, md.grid, t, pz; method = md.method, vdirs = (pz,))
    md.dist.data .+= jac
    return sp
end

function push_field!(sp::LinearDKSpecies, E::VectorField, grid::Grid, dt::Real; phase = 0)
    md = getfield(sp, :model)
    pz = grid.Bdir
    add_linear_dk_kick!(md.dist, md.grid, dt, E[pz], curl(E, grid)[pz];
                        q = sp.q, m = sp.m, T_e = md.T_e, c0 = md.c0, c1 = md.c1)
    return sp
end

# Charge density q ∫δF and the parallel current q (1/√m) ∫v δF at index Bdir.
function moments_at(sp::LinearDKSpecies, grid::Grid, phase::Real)
    md = getfield(sp, :model)
    n = compute_density(md.dist, md.grid).data
    J = sp.q * thermal_velocity(sp) .* compute_current(md.dist, md.grid)[1].data
    z = zero(n)
    return (sp.q .* n, ntuple(d -> d == grid.Bdir ? J : z, 3))
end

# Free energy ∫ T_e δf² / (2 F0), δf = δF − F0 δB∥/b0; its rate is ∫ E∥ J∥.
function species_energy(sp::LinearDKSpecies, grid::Grid, B)
    md = getfield(sp, :model)
    nx = length(md.grid.xaxes)
    F0 = reshape(_dk_maxwellian(md)[1], ntuple(_ -> 1, nx)..., :)
    δf = md.dist.data .- (B[grid.Bdir].data ./ grid.b0) .* F0
    return prod(md.grid.delta) * md.T_e / 2 * sum(δf .^ 2 ./ F0)
end

_save(sp::LinearDKSpecies) = copy(sp.dist.data)
_restore!(sp::LinearDKSpecies, s) = (sp.dist.data .= s; sp)
_stage_phase(::LinearDKSpecies, grid, dt, phase) = phase
_end_phase(::LinearDKSpecies, grid, dt, phase) = phase

response_class(::LinearDK) = Drift()

drift_density_hat(sp::LinearDKSpecies, sw) =
    _fft_c(compute_density(sp.dist, getfield(sp, :model).grid).data, sw)

# Set the density to N̂ by adding the change times F0 (the shape of a density perturbation).
function set_drift_density!(sp::LinearDKSpecies, N, sw)
    md = getfield(sp, :model)
    n_old = compute_density(md.dist, md.grid).data
    n_new = similar(n_old)
    ifft_spatial!(n_new, N, sw)
    F0 = _dk_maxwellian(md)[1]
    nx = length(md.grid.xaxes)
    md.dist.data .+= (n_new .- n_old) .* reshape(F0, ntuple(_ -> 1, nx)..., length(F0))
    return sp
end

# Stage response (drift-fluid form). With the streaming multipliers g = exp(-i k∥ v dt/(2√m)),
# J^θ = q E×b̂/b0 + q (U + θ dt (q/m) E∥) b̂ with U = ∫ v δF^a / √m, and, since the second half
# step streams δF − F0 B∥^{n+1}/b0 and B∥^{n+1} = B∥^n − dt b̂·(ik×E),
# ρ^{n+1} = q [Σ g δF^a Δv + (1 − G0) B∥^n/b0 - dt b̂·(ik×E)/b0 - dt (q/√m) G1 E∥],
# G0 = Σ g F0 Δv, G1 = Σ g F0' Δv.
# Exact for Fourier streaming; the Lagrange streaming gets the corrector.
function stage_response(sp::LinearDKSpecies, grid::Grid, dt, θ; phase = 0, B = nothing)
    B === nothing && throw(ArgumentError("LinearDK stage response needs B^n"))
    md = getfield(sp, :model)
    a, b, pz = _cf_frame(grid)
    q, m, b0 = sp.q, sp.m, grid.b0
    ρ, J = moments_at(sp, grid, phase)
    sw, ka, kb, kz = _cf_kviews(ρ, grid)
    F(x) = _fft_c(x, sw)
    U = F(J[pz]) ./ q
    Z = zero(U)
    Nd, G0, G1 = zero(U), zero(U), zero(U)
    F0, dF0 = _dk_maxwellian(md)
    v, dv, vth = collect(md.grid.vaxes[1]), _dk_dv(md), thermal_velocity(sp)
    nx = length(md.grid.xaxes)
    for j in eachindex(v)
        Fj = F(copy(selectdim(md.dist.data, nx + 1, j)))
        @. Nd += dv * cis(-kz * vth * v[j] * dt / 2) * Fj
        @. G0 += dv * cis(-kz * vth * v[j] * dt / 2) * F0[j]
        @. G1 += dv * cis(-kz * vth * v[j] * dt / 2) * dF0[j]
    end
    c(x) = fill!(similar(Z), x)
    J = ((Z, Z, c(q / b0), Z), (Z, c(-q / b0), Z, Z), (q .* U, Z, Z, c(θ * dt * q^2 / m)))
    Bz = F(B[pz].data)
    rho = (@.(q * (Nd + (1 - G0) * Bz / b0)), @.(q * dt / b0 * im * kb + 0 * Z), @.(-q * dt / b0 * im * ka + 0 * Z),
           @.(-q * dt * q * vth * G1))
    return (J = J, rho = rho, Jp = (_affine(Z), _affine(Z)), drift = (q, m),
            exact = md.method isa Fourier)
end
