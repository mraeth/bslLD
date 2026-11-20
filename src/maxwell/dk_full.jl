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
# (`exb = :rk4`, `exb_rk4!`; stable for k_max|u_E|Δt ≤ 2.8), optionally dealiased (RK4 only): by the 2/3 Galerkin
# truncation that keeps the bracket skew-symmetric (`dealias = :truncate` or `true`), or by 3/2 padding, which keeps all
# modes (`dealias = :pad`). In the charge-form step the
# kick is applied in halves around the E×B push (`_push_stage!`), second order in their commutator.
# A drift species: E×B density response and polarisation density in the field solve.
# Walls along x1 (`x_boundary = :mirror`, the `Mirror()` grid with walls on the first and last node): the E×B push
# runs on the extension of F (even) and φ (odd, φ = 0 on the walls) to the periodic grid of 2(N−1) nodes and is
# restricted back; u_x = −∂_yφ/b0 vanishes on the walls, so nothing crosses them. Kick and streaming are pointwise
# in x1 and unchanged.

struct FullDK{DG<:DistributionGrid,G<:Grid,MT,EX} <: DistributionModel
    dist::DG      # F on `grid`
    grid::G       # x axes of the field grid, one v∥ axis
    T_e::Vector{Float64}  # electron temperature along x1 (the Maxwellian of the initial state and of G(k∥))
    method::MT    # Fourier() or Lagrange(order) for the parallel streaming
    exb::Symbol   # :euler or :rk4 E×B push
    dealias::Symbol # :none, :truncate (2/3) or :pad (3/2 padding) of the :rk4 bracket
    ext::EX       # nothing (periodic x1), or (grid, F buffer) of the extended x1 grid with walls
end

"""
    FullDK(field_grid, n; T_e, Nv = 64, vmax = 6, method = Fourier(), exb = :euler, dealias = false,
           x_boundary = :periodic)

Full-f drift-kinetic model on the x axes of `field_grid` with a v∥ axis of `Nv` intervals (`Nv + 1`
nodes) on ±`vmax` √max T_e, initialised as n(x) F_M(v∥; T_e(x1)) with the density `n` (an array of the spatial
shape) and a Maxwellian normalised on the grid at every x1. `T_e`: a number, a vector along x1, or a function of x1. Species: `Species(FullDK(grid, n; T_e); q = -1, m = μ)`.
The E×B advection needs b̂ = x̂3 (`field_grid.Bdir == 3`); `exb` is its time integrator over the
step, `:euler` (one stage) or `:rk4`; `dealias` (`:rk4` only): `:truncate` (or `true`) the 2/3 Galerkin truncation,
`:pad` the 3/2-padded bracket, `:none` (or `false`).
`x_boundary = :mirror`: walls on the first and last x1 node (pair with `ESChargeForm(x_boundary = :dirichlet)`).
"""
function FullDK(field_grid::CartGrid, n::AbstractArray; T_e, Nv::Int = 64, vmax::Real = 6,
                method = Fourier(), exb::Symbol = :euler, dealias::Union{Bool,Symbol} = false,
                x_boundary::Symbol = :periodic)
    exb in (:euler, :rk4) || throw(ArgumentError("exb must be :euler or :rk4, got :$exb"))
    x_boundary in (:periodic, :mirror) || throw(ArgumentError("x_boundary must be :periodic or :mirror"))
    dmode = _exb_mode(dealias)
    dmode !== :none && exb !== :rk4 && throw(ArgumentError("dealias needs exb = :rk4"))
    x1 = collect(field_grid.xaxes[1])
    Te = T_e isa Real ? fill(Float64(T_e), length(x1)) : T_e isa AbstractVector ? Vector{Float64}(T_e) :
         Float64.(T_e.(x1))
    length(Te) == length(x1) || throw(ArgumentError("T_e has $(length(Te)) values, the x1 axis $(length(x1))"))
    all(>(0), Te) || throw(ArgumentError("T_e must be positive"))
    field_grid.Bdir == 3 || throw(ArgumentError("FullDK needs b̂ = x̂3 (Bdir = 3) for the E×B bracket"))
    nx = length(field_grid.xaxes)
    size(n) == Tuple(length.(field_grid.xaxes)) ||
        throw(ArgumentError("density of size $(size(n)), grid $(Tuple(length.(field_grid.xaxes)))"))
    vm = vmax * sqrt(maximum(Te))
    grid = Grid(vcat(Vector{Float64}(field_grid.min[1:nx]), -vm),
                vcat(Vector{Float64}(field_grid.max[1:nx]), vm),
                vcat(collect(length.(field_grid.xaxes)), Nv), nx, field_grid.b0, field_grid.Bdir)
    v, dv = collect(grid.vaxes[1]), grid.delta[nx+1]
    g = exp.(-reshape(v, 1, :) .^ 2 ./ (2 .* Te))          # (x1, v∥)
    g ./= dv .* sum(g; dims = 2)
    sf = ScalarField(backend_array(Float64.(n) .* reshape(g, length(x1), ntuple(_ -> 1, nx - 1)..., :)))
    dist = DistributionGridImpl{nx,1,Cart,typeof(sf)}(sf)
    ext = nothing
    if x_boundary === :mirror
        N1, h = length(grid.xaxes[1]), grid.delta[1]
        xmax = copy(Vector{Float64}(grid.max))
        xmax[1] = grid.min[1] + (2N1 - 2) * h
        egrid = Grid(Vector{Float64}(grid.min), xmax, vcat(2N1 - 2, collect(length.(field_grid.xaxes))[2:nx], Nv), nx,
                     field_grid.b0, field_grid.Bdir)
        esf = ScalarField(similar(sf.data, Float64, (2N1 - 2, size(sf.data)[2:end]...)))
        ext = (egrid, DistributionGridImpl{nx,1,Cart,typeof(esf)}(esf))
    end
    return FullDK(dist, grid, Te, method, exb, dmode, ext)
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
    adv!(f, ϕ, g) = md.exb === :rk4 ? exb_rk4!(f, ϕ, g, dt; md.dealias) : exb_euler!(f, f, ϕ, g, dt)
    if md.ext === nothing
        adv!(md.dist, φ, md.grid)
    else
        egrid, ef = md.ext
        N1 = size(md.dist.data, 1)
        ef.data .= _x1_extend(md.dist.data, 1)
        adv!(ef, ScalarField(_x1_extend(φ.data, -1)), egrid)
        md.dist.data .= selectdim(ef.data, 1, 1:N1)
    end
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

# Whether the E×B push of a drift species is 2/3-dealiased or 3/2-padded (the stage operator then matches it).
_exb_dealiased(sp) = false
_exb_dealiased(sp::FullDKSpecies) = getfield(sp, :model).dealias === :truncate
_exb_padded(sp) = false
_exb_padded(sp::FullDKSpecies) = getfield(sp, :model).dealias === :pad
