# Full-f drift-kinetic species (no δf split).
#
# State F(x, v∥) on its own 1V grid (the x axes of the field grid and one v∥ axis, in units of 1/√m),
# no perpendicular temperature (no mirror force), uniform B0 = b0 b̂ with b̂ = x̂3. Guiding-centre
# Vlasov equation
#
#   ∂t F + u_E·∇F + v ∂∥F + (q/√m) E∥ ∂_v F = 0,   u_E = E×b̂/b0 = b̂×∇φ/b0.
#
# Free part (`push_free!`): parallel streaming (`advectX!` along Bdir, `vdirs = (Bdir,)`). Field part:
# the parallel kick (`push_field!`, `advectV!`) and the E×B advection (`push_drift!`) of the spectral
# bracket with φ frozen over the step: one Euler step (`exb = :euler`, `exb_euler!`), exact at linear
# order about an x-dependent F but first order in the E×B nonlinearity, or classical RK4
# (`exb = :rk4`, `exb_rk4!`; stable for k_max|u_E|Δt ≤ 2.8), optionally dealiased by the 2/3 Galerkin
# truncation that keeps the bracket skew-symmetric (`dealias = true`, RK4 only). In the charge-form step the
# kick is applied in halves around the E×B push (`_push_stage!`), second order in their commutator.
# A drift species: E×B density response and polarisation density in the field solve.

struct FullDK{DG<:DistributionGrid,G<:Grid,MT} <: DistributionModel
    dist::DG      # F on `grid`
    grid::G       # x axes of the field grid, one v∥ axis
    T_e::Float64
    method::MT    # Fourier() or Lagrange(order) for the parallel streaming
    exb::Symbol   # :euler or :rk4 E×B push
    dealias::Bool # 2/3 truncation of the :rk4 bracket
end

"""
    FullDK(field_grid, n; T_e, Nv = 64, vmax = 6, method = Fourier(), exb = :euler, dealias = false)

Full-f drift-kinetic model on the x axes of `field_grid` with a v∥ axis of `Nv` intervals (`Nv + 1`
nodes) on ±`vmax` √T_e, initialised as n(x) F_M(v∥) with the density `n` (an array of the spatial
shape) and a Maxwellian normalised on the grid. Species: `Species(FullDK(grid, n; T_e); q = -1, m = μ)`.
The E×B advection needs b̂ = x̂3 (`field_grid.Bdir == 3`); `exb` is its time integrator over the
step, `:euler` (one stage) or `:rk4`; `dealias` applies the 2/3 Galerkin truncation to the `:rk4` bracket.
"""
function FullDK(field_grid::CartGrid, n::AbstractArray; T_e::Real, Nv::Int = 64, vmax::Real = 6,
                method = Fourier(), exb::Symbol = :euler, dealias::Bool = false)
    exb in (:euler, :rk4) || throw(ArgumentError("exb must be :euler or :rk4, got :$exb"))
    dealias && exb !== :rk4 && throw(ArgumentError("dealias needs exb = :rk4"))
    T_e > 0 || throw(ArgumentError("T_e must be positive"))
    field_grid.Bdir == 3 || throw(ArgumentError("FullDK needs b̂ = x̂3 (Bdir = 3) for the E×B bracket"))
    nx = length(field_grid.xaxes)
    size(n) == Tuple(length.(field_grid.xaxes)) ||
        throw(ArgumentError("density of size $(size(n)), grid $(Tuple(length.(field_grid.xaxes)))"))
    vm = vmax * sqrt(T_e)
    grid = Grid(vcat(Vector{Float64}(field_grid.min[1:nx]), -vm),
                vcat(Vector{Float64}(field_grid.max[1:nx]), vm),
                vcat(collect(length.(field_grid.xaxes)), Nv), nx, field_grid.b0, field_grid.Bdir)
    v, dv = collect(grid.vaxes[1]), grid.delta[nx+1]
    g = @. exp(-v^2 / (2T_e))
    g ./= dv * sum(g)
    sf = ScalarField(backend_array(Float64.(n) .* reshape(g, ntuple(_ -> 1, nx)..., :)))
    dist = DistributionGridImpl{nx,1,Cart,typeof(sf)}(sf)
    return FullDK(dist, grid, Float64(T_e), method, exb, dealias)
end

const FullDKSpecies = Species{<:AbstractFloat,<:FullDK}

function push_free!(sp::FullDKSpecies, grid::Grid, dt::Real, half::Int; phase = 0, B = nothing)
    md = getfield(sp, :model)
    pz = md.grid.Bdir
    pz <= length(md.grid.xaxes) || return sp      # no parallel axis: k∥ = 0
    t = SimulationTime(dt, dt)
    t.fraction_dt = 0.5
    advectX!(sp, md.grid, t, pz; method = md.method, vdirs = (pz,))
    return sp
end

function push_field!(sp::FullDKSpecies, E::VectorField, grid::Grid, dt::Real; phase = 0)
    md = getfield(sp, :model)
    pz = md.grid.Bdir
    pz <= length(md.grid.xaxes) || return sp      # E∥ = 0 without a parallel axis
    advectV!(sp, md.grid, SimulationTime(dt, dt), E; vdirs = (pz,))
    return sp
end

"""
    push_drift!(sp, φ::ScalarField, grid, dt)

E×B advection of a drift species by the potential `φ` over `dt` (no-op for other species).
"""
push_drift!(sp, φ::ScalarField, grid::Grid, dt::Real) = sp

function push_drift!(sp::FullDKSpecies, φ::ScalarField, grid::Grid, dt::Real)
    md = getfield(sp, :model)
    md.exb === :rk4 ? exb_rk4!(md.dist, φ, md.grid, dt; md.dealias) : exb_euler!(md.dist, md.dist, φ, md.grid, dt)
    return sp
end

# Charge density q ∫F and the parallel current q (1/√m) ∫v F at index Bdir.
function moments_at(sp::FullDKSpecies, grid::Grid, phase::Real)
    md = getfield(sp, :model)
    n = compute_density(md.dist, md.grid).data
    J = sp.q * thermal_velocity(sp) .* compute_current(md.dist, md.grid)[1].data
    z = zero(n)
    return (sp.q .* n, ntuple(d -> d == grid.Bdir ? J : z, 3))
end

_save(sp::FullDKSpecies) = copy(sp.dist.data)
_restore!(sp::FullDKSpecies, s) = (sp.dist.data .= s; sp)
_stage_phase(::FullDKSpecies, grid, dt, phase) = phase
_end_phase(::FullDKSpecies, grid, dt, phase) = phase

response_class(::FullDK) = Drift()

# Whether the E×B push of a drift species is 2/3-dealiased (the stage operator then matches it).
_exb_dealiased(sp) = false
_exb_dealiased(sp::FullDKSpecies) = getfield(sp, :model).dealias
