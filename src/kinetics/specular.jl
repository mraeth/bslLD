# Specular wall on a spatial axis perpendicular to B, on the rotating logical grid.
#
# Physical specular reflection on axis `dir` flips v_dir only. With v = Q u,
# Q = R(-φ), φ = electric_scale * phase, the logical map is
#
#     u' = Q^T S Q u = R(2φ + δ) F u,     F = diag(-1, 1),  δ = 0 (dir 1) or π (dir 2),
#
# a u_x index flip followed by a rotation in (u_x, u_y). `Mirror()` flips both
# u_x and u_y instead (u' = -u), which is rotation invariant but not a symmetry of
# a mode with structure along the wall (the guiding centre y + v_x/Ω changes).
#
# The halo is filled node-fold, as for Mirror (ghost 1-k <- node 1+k, ghost n+k
# <- node n-k), but from interior planes remapped in velocity: the rotation is
# reduced to |θ| <= π/2 by an exact index rotation by π and done as three
# Lagrange shears, applied to the halo planes only.

"""
    Specular()

Specular reflection (`v_dir -> -v_dir`) at the first and last node of a spatial
axis perpendicular to `B` (`grid.Bdir == 3`), with node-fold halo and wall-node
symmetrisation `f(x_b, u) = f(x_b, u')`. On the axis along `B` it is identical to
[`Mirror`](@ref). Needs a `Lagrange` method and velocity axes symmetric about zero.
"""
struct Specular end

struct SpecularHalo{A,SXT}
    lo::A        # lo[.., k, .., flip(u)] = f at ghost position 1 - k
    hi::A        # hi[.., k, .., flip(u)] = f at ghost position n + k
    sizes_x::SXT # x-sizes of the slabs (axis `dir` has length D)
    D::Int
    flipdim::Int # velocity axis whose index is reversed on reading
end

Adapt.@adapt_structure SpecularHalo

@inline function _sample(h::SpecularHalo, src, ctx::XShiftContext, ixs, ivs, i, n)
    if i < 1 || i > n
        fv = Base.setindex(ivs, ctx.sizes_v[h.flipdim] + 1 - ivs[h.flipdim], h.flipdim)
        k = i < 1 ? 1 - i : i - n
        J = index_combined_to_1d(Base.setindex(ixs, k, ctx.dir), fv, h.sizes_x, ctx.sizes_v)
        return @inbounds (i < 1 ? h.lo[J] : h.hi[J])
    end
    return @inbounds src[_line_linear_index(ctx, ixs, ivs, i)]
end

# Planes `idx` of `data` along spatial axis `dir`, as a new array.
_planes(data, dir, idx) = data[ntuple(d -> d == dir ? idx : Colon(), ndims(data))...]

# g(u) = slab(T u), T = R(θ) F: rotate by the reduced angle with three shears. The
# index permutation that remains (u_x flip, or u_y flip after the π rotation) is
# returned as the velocity axis to reverse: g(u) = rotated[flip(u)].
function _specular_remap(slab, ctx::XShiftContext, method, exec)
    NX = length(ctx.sizes_x)
    DT = eltype(slab)
    theta = 2 * ctx.electric_scale * ctx.phi + (ctx.dir == 2 ? DT(pi) : zero(DT))
    k = round(Int, theta / pi)
    theta -= k * DT(pi)
    sizes_x = ntuple(d -> size(slab, d), NX)
    grid = ctx.grid
    dv1, dv2 = DT(grid.delta[NX+1]), DT(grid.delta[NX+2])
    na = sizes_x[1]
    a = fill!(similar(slab, DT, na), tan(theta / 2) / dv1)
    b = fill!(similar(slab, DT, na), -sin(theta) / dv2)
    b1, b2 = similar(slab), similar(slab)            # `slab` itself is left intact
    shear!(dst, src, dir, s) = _lagrange_sweep!(dst, src,
        LineShiftContext{:v}(sizes_x, ctx.sizes_v, dir, s), method, Periodic(), exec)
    sx = VelocityShear(a, ctx.vaxes[2], 2)
    shear!(b1, slab, 1, sx)
    shear!(b2, b1, 2, VelocityShear(b, ctx.vaxes[1], 1))
    shear!(b1, b2, 1, sx)
    # R(kπ) F = F (k even) or diag(1, -1) (k odd)
    return b1, iseven(k) ? 1 : 2
end

# Halo depth covering the widest stencil of this sweep.
_specular_depth(ctx::XShiftContext, ::Val{W}) where {W} =
    min(ctx.sizes_x[ctx.dir] - 1, floor(Int, _max_shift_cells(ctx)) + W ÷ 2 + 2)

function _check_specular(ctx::XShiftContext)
    ctx.grid.Bdir == 3 && length(ctx.sizes_v) >= 2 || throw(ArgumentError(
        "Specular needs B along axis 3 and at least two velocity components"))
    return nothing
end

function _advect_dir!(f, ctx::XShiftContext, plan, method::Lagrange, ::Specular, fwd, inv)
    ctx.dir == ctx.grid.Bdir && return _advect_dir!(f, ctx, plan, method, Mirror(), fwd, inv)
    _check_specular(ctx)
    exec = plan.backend
    n = ctx.sizes_x[ctx.dir]
    dir = ctx.dir

    # wall nodes: f(x_b, u) <- (f(x_b, u) + f(x_b, T u)) / 2
    for i in (1, n)
        w = _planes(f.data, dir, i:i)
        sel = ntuple(d -> d == dir ? (i:i) : Colon(), ndims(f.data))
        r, fd = _specular_remap(w, ctx, method, exec)
        f.data[sel...] .= (w .+ reverse(r; dims = length(ctx.sizes_x) + fd)) ./ 2
    end

    D = _specular_depth(ctx, stencil_order(method))
    lo, fd = _specular_remap(_planes(f.data, dir, 2:(D+1)), ctx, method, exec)
    hi, _ = _specular_remap(_planes(f.data, dir, (n-1):-1:(n-D)), ctx, method, exec)
    halo = SpecularHalo(lo, hi, Base.setindex(ctx.sizes_x, D, dir), D, fd)

    _lagrange_sweep!(plan.rbuf, f.data, ctx, method, halo, exec)
    copyto!(f.data, plan.rbuf)
    return nothing
end
