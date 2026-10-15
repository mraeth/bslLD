# Lagrange interpolation of arbitrary (even) order on a uniform grid.
#
# For a backward-semi-Lagrangian shift the departure point of grid node j is
# x_j - s*h. Splitting s into an integer cell count and a fraction alpha in
# [0,1) (see `shift_split`) leaves a single interpolation problem: evaluate a
# uniform-grid function at -alpha cells from a node. The centred stencil of
# even width W covers offsets -W/2 ... W/2-1, which brackets [-1,0] for every W.
#
# Weights are computed in O(W) via prefix/suffix products of the node differences,
# with the binomial factors by recurrence and one final normalisation:
#
#   w_m = c_m * prod_{k != m} (u - k),   u = W/2 - alpha,  k = 0 ... W-1
#
# with c_m the (alternating, binomial) reciprocal node-difference product. The
# product form has no pole when the departure point lands exactly on a node --
# unlike the barycentric formula, which divides by (u - k) -- so alpha = 0 and
# alpha = 1 fall out as clean delta stencils.
#
# Accumulation is in Float64 regardless of the requested element type: the
# intermediate products span ~(W/2)^(W-1), which overflows Float32 beyond
# W ~ 26. Weights are needed once per distinct shift (once per line for
# constant-shift advection), never once per point, so this costs nothing in the
# inner loop.

"""
    lagrange_weights(Val(W), alpha, T = Float64) -> SVector{W,T}

Weights of the centred Lagrange stencil of even width `W` for evaluating a
uniform-grid function at `alpha` cells *below* node `j`, `0 <= alpha <= 1`:

    f(x_j - alpha*h) ≈ sum_m w[m] * f[j + m - 1 - W÷2]      (m = 1 ... W)

`sum(w) == 1` to roundoff by construction. Allocation-free and GPU-safe.
"""
@inline function lagrange_weights(::Val{W}, alpha, ::Type{T} = Float64) where {W,T}
    iseven(W) || throw(ArgumentError("Lagrange stencil width must be even, got $W"))
    u = W ÷ 2 - Float64(alpha)                      # departure point, stencil-local
    @inline diff(m) = u - (m - 1)                   # u - k for k = m-1

    suffix = MVector{W,Float64}(undef)              # suffix[m] = prod_{k>m-1} (u-k)
    acc = 1.0
    @inbounds for m = W:-1:1
        suffix[m] = acc
        acc *= diff(m)
    end

    v = MVector{W,Float64}(undef)
    prefix = 1.0                                    # prod_{k<m-1} (u-k)
    binom = 1.0                                     # binomial(W-1, m-1), by recurrence
    total = 0.0
    @inbounds for m = 1:W
        vm = (iseven(W - m) ? binom : -binom) * prefix * suffix[m]
        v[m] = vm
        total += vm
        prefix *= diff(m)
        binom *= (W - m) / m
    end

    inv_total = 1 / total                           # == 1/(W-1)!, normalises sum(w) to 1
    return SVector{W,T}(ntuple(m -> T(@inbounds(v[m]) * inv_total), Val(W)))
end

"""
    shift_split(s) -> (cells, alpha)

Split a displacement of `s` cells into an integer cell count and a fraction
`alpha` in `[0,1)`, so that `x_j - s*h` is `alpha` cells below node `j - cells`.
"""
@inline function shift_split(s::Real)
    cells = floor(Int, s)
    return cells, s - cells
end
