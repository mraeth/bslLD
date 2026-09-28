# Spectral response of 1D backward-semi-Lagrangian interpolation kernels:
# what each scheme does to the k-spectrum, how to prescribe that response
# instead of inheriting it, and why a spline is a convolution stencil too.
#
# Every scheme here is linear and translation invariant on a uniform grid, so a
# shift by alpha*h maps a mode exp(i*k*x_j) to g(theta)*exp(i*k*x_j) with
# theta = k*h in [0, pi]. The exact shift is exp(-i*alpha*theta), so |g| < 1 is
# numerical damping, arg(g) + alpha*theta is the dispersion error, and |g| > 1
# anywhere in the band means the scheme amplifies.
#
# Run: julia --project=examples examples/interpolation_spectra.jl

using Printf, LinearAlgebra, FFTW, CairoMakie

const ALPHA = 0.3                              # shift in cells
const NSTEP = 200                              # steps for the "surviving spectrum" diagnostic
const ALPHAS = collect(range(0, 1, length = 21)[1:end-1])   # stability scan
const THETAS = range(0, pi, length = 600)

# --- FIR machinery --------------------------------------------------------

offsets(w) = -(length(w) ÷ 2):(length(w) - 1) ÷ 2

fir_symbol(w, theta) = sum(wi * cis(theta * l) for (wi, l) in zip(w, offsets(w)))

"Centered Lagrange weights of even width W for evaluation at -alpha."
function lagrange_weights(W::Int, alpha::Real)
    off = -W÷2:W÷2-1
    [prod((-alpha - m) / (l - m) for m in off if m != l) for l in off]
end

lagrange_symbol(W, alpha, theta) = fir_symbol(lagrange_weights(W, alpha), theta)

conv1(a, b) = [sum(a[i] * b[k-i+1] for i = max(1, k - length(b) + 1):min(length(a), k))
               for k = 1:length(a)+length(b)-1]

# --- splines --------------------------------------------------------------

"Cardinal B-spline of degree p, centered on 0 (Unser's recursion)."
function bspline(p::Int, x::Real)
    p == 0 && return abs(x) < 0.5 ? 1.0 : (abs(x) == 0.5 ? 0.5 : 0.0)
    ((x + (p + 1) / 2) * bspline(p - 1, x + 0.5) +
     ((p + 1) / 2 - x) * bspline(p - 1, x - 0.5)) / p
end

"""
Spline interpolation = IIR prefilter (deconvolution of the B-spline) followed
by FIR evaluation, so the symbol is RATIONAL in exp(i*theta) -- that is the
only structural difference to a plain stencil.
"""
function spline_symbol(p::Int, alpha::Real, theta::Real)
    ms = -(p + 1):(p + 1)
    sum(bspline(p, m - alpha) * cis(-theta * m) for m in ms) /
    sum(bspline(p, m) * cis(-theta * m) for m in ms)
end

"""
The spline written as a convolution: Fourier coefficients of its symbol, i.e.
the (infinite, geometrically decaying) equivalent kernel. Truncating it to W
taps is the local-spline / LSL construction.
"""
const NQ = 2048
function spline_coeffs(p::Int, alpha::Real)
    ms = collect(-(p + 1):(p + 1))
    bn = [bspline(p, m - alpha) for m in ms]          # hoist the recursion out of the loop
    bd = [bspline(p, m) for m in ms]
    G = map(2pi .* (0:NQ-1) ./ NQ) do t
        e = [cis(-t * m) for m in ms]
        (bn' * e) / (bd' * e)
    end
    real.(fft(G) ./ NQ)                               # c[l+1] <-> exp(+i*l*theta)
end

truncated_spline_weights(c, W) = (k = [c[mod(l, NQ)+1] for l = -W÷2:W÷2-1]; k ./ sum(k))

# --- route A: fit the shift weights to a prescribed response --------------
#
# Ask for g(theta) = T(theta)*exp(-i*alpha*theta) directly and solve for the W
# weights in least squares, constrained to reproduce constants and linears.
# Ordinary FIR design -- but a finite stencil leaves Gibbs ripple around the
# target, and the ripple sits ABOVE 1 in the flat part of the band.
# Note T(pi) must vanish: a real stencil has real g(pi), so the exact shift of
# the Nyquist mode is unrepresentable and every scheme here kills it.

function fit_weights(W::Int, alpha::Real, T; nq = 400)
    off = collect(-W÷2:W÷2-1)
    th = range(0, pi, length = nq)
    A = [cos.(th * off'); sin.(th * off')]
    g = [T(t) * cis(-alpha * t) for t in th]
    b = [real.(g); imag.(g)]
    C, d = [ones(W)'; off'], [1.0, -alpha]     # sum(w)=1, sum(l*w)=-alpha
    ([2A'A C'; C zeros(2, 2)] \ [2A'b; d])[1:W]
end

"Flat up to theta0, raised-cosine roll-off to 0 at pi."
target(theta; theta0 = 0.5pi) =
    theta <= theta0 ? 1.0 : 0.5 * (1 + cos(pi * (theta - theta0) / (pi - theta0)))

# --- route B: put the damping in a separate symmetric filter --------------
#
# g_total = g_interp * g_filter with
#     g_f(theta) = 1 - eps * sin^{2n}(theta/2),
# a trigonometric polynomial of degree n, i.e. a symmetric (2n+1)-point
# stencil, with 0 <= g_f <= 1 BY CONSTRUCTION: no fitting, no ripple, exact on
# the constant mode, zero phase error, and independent of alpha.
# sin^2(theta/2) is the symbol of [-1/4, 1/2, -1/4], so n convolutions of that
# 3-tap kernel give the weights; the filter is the operator I - eps*(-D2/4)^n.

function flat_filter(n::Int; eps::Real = 1.0)
    w = [1.0]
    for _ = 1:n
        w = conv1(w, [-0.25, 0.5, -0.25])
    end
    w .*= -eps
    w[n+1] += 1.0
    w                                          # offsets -n:n
end

# --- schemes --------------------------------------------------------------

const w_fit = fit_weights(12, ALPHA, target)
const w_filt = flat_filter(8, eps = 0.0128)    # eps set from the CUMULATIVE target

const SCHEMES = [
    ("LAG8", theta -> lagrange_symbol(8, ALPHA, theta)),
    ("LAG24", theta -> lagrange_symbol(24, ALPHA, theta)),
    ("spline3", theta -> spline_symbol(3, ALPHA, theta)),
    ("spline5", theta -> spline_symbol(5, ALPHA, theta)),
    ("spline7", theta -> spline_symbol(7, ALPHA, theta)),
    ("fit12", theta -> fir_symbol(w_fit, theta)),
    ("LAG24+filt", theta -> lagrange_symbol(24, ALPHA, theta) * fir_symbol(w_filt, theta)),
]

# --- sanity check: symbols vs. the operator applied to a grid mode --------

let theta = 0.4pi, n = 256
    xs = (0:n-1) .* (2pi / n)
    k = theta * n / 2pi
    w, off = lagrange_weights(8, ALPHA), collect(-4:3)
    apply(f) = [sum(w[i] * f[mod1(j + off[i], n)] for i in eachindex(off)) for j = 1:n]
    j = n ÷ 4 + 1
    c, s = apply(cos.(k .* xs)), apply(sin.(k .* xs))
    @printf("symbol check (LAG8): |analytic - measured| = %.2e\n", abs(
        (c[j] + im * s[j]) / cis(k * xs[j]) - lagrange_symbol(8, ALPHA, theta)))
end

# --- tables ---------------------------------------------------------------

header(title) = (println("\n", title, "\n");
                 print(@sprintf("%7s", "kh/pi"));
                 foreach(s -> print(@sprintf("%12s", s[1])), SCHEMES); println())

header("|g| per step (alpha = $ALPHA)")
for f in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    print(@sprintf("%7.3f", f))
    foreach(s -> print(@sprintf("%12.6f", abs(s[2](f * pi)))), SCHEMES)
    println()
end

header("phase error [rad]")
for f in (0.25, 0.375, 0.5, 0.625, 0.75)
    print(@sprintf("%7.3f", f))
    foreach(s -> print(@sprintf("%12.1e",
        angle(s[2](f * pi) / cis(-ALPHA * f * pi)))), SCHEMES)
    println()
end

header("amplitude retained after $NSTEP steps")
for f in (0.25, 0.5, 0.625, 0.75, 0.875)
    print(@sprintf("%7.3f", f))
    foreach(s -> print(@sprintf("%12.4f", abs(s[2](f * pi))^NSTEP)), SCHEMES)
    println()
end

println("\nstability, max |g| over the band (> 1 means the scheme amplifies):")
for (name, sym) in SCHEMES
    gmax = maximum(abs(sym(t)) for t in range(0, pi, length = 4000))
    @printf("  %-11s %.6f  %s\n", name, gmax, gmax > 1 + 1e-9 ? "<-- AMPLIFYING" : "")
end

println("\nprescribed filter family  g_f = 1 - sin^2n(theta/2)  (route B, eps = 1):\n")
@printf("%12s", "n (width)")
foreach(f -> @printf("%9.3f", f), (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
println("     min       max")
for n in (4, 6, 8, 10, 12)
    w = flat_filter(n)
    @printf("%6d (%2d)", n, 2n + 1)
    foreach(f -> @printf("%9.4f", real(fir_symbol(w, f * pi))),
            (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
    @printf("  %8.4f  %.6f\n",
            minimum(real(fir_symbol(w, t)) for t in THETAS),
            maximum(abs(fir_symbol(w, t)) for t in THETAS))
end

# --- a spline IS a stencil: equal-width comparison ------------------------

const C = Dict((p, a) => spline_coeffs(p, a) for p in (3, 5, 7), a in ALPHAS)

function scan(mk)                              # (|g| at 0.5pi, at 0.625pi, max|g| over alpha)
    w = mk(ALPHA)
    (abs(fir_symbol(w, 0.5pi)), abs(fir_symbol(w, 0.625pi)),
     maximum(maximum(abs(fir_symbol(mk(a), t)) for t in THETAS) for a in ALPHAS))
end
cell(t) = @sprintf("%.5f %.5f %s", t[1], t[2], t[3] > 1 + 1e-9 ? @sprintf("A%.4f", t[3]) : "ok     ")

println("\nthe spline as a truncated convolution (= LSL).")
println("|g| at kh = 0.5pi / 0.625pi, then max|g| over alpha in [0,1):\n")
@printf("%4s | %-22s | %-22s | %-22s\n", "W", "LAG W", "spline5 trunc. to W", "spline7 trunc. to W")
for W in (8, 12, 16, 24, 32)
    @printf("%4d | %s | %s | %s\n", W,
            cell(scan(a -> lagrange_weights(W, a))),
            cell(scan(a -> truncated_spline_weights(C[(5, a)], W))),
            cell(scan(a -> truncated_spline_weights(C[(7, a)], W))))
end
@printf("%4s | %22s | %.5f %.5f        | %.5f %.5f\n", "IIR", "(untruncated spline)",
        abs(spline_symbol(5, ALPHA, 0.5pi)), abs(spline_symbol(5, ALPHA, 0.625pi)),
        abs(spline_symbol(7, ALPHA, 0.5pi)), abs(spline_symbol(7, ALPHA, 0.625pi)))

println("\nequivalent-kernel decay |c_l| (alpha = $ALPHA):\n")
@printf("%6s %11s %11s %11s\n", "l", "spline3", "spline5", "spline7")
for l in (4, 8, 12, 16, 20, 24, 32)
    @printf("%6d %11.2e %11.2e %11.2e\n", l,
            abs(C[(3, 0.3)][l+1]), abs(C[(5, 0.3)][l+1]), abs(C[(7, 0.3)][l+1]))
end

# --- figure ---------------------------------------------------------------

th = range(1e-3, pi, length = 400)
fig = Figure(size = (1100, 780))
ax1 = Axis(fig[1, 1], xlabel = "k h / π", ylabel = "|g| per step",
           title = "damping per shift (α = $ALPHA)")
ax2 = Axis(fig[1, 2], xlabel = "k h / π", ylabel = "|g|^$NSTEP",
           title = "spectrum surviving $NSTEP steps")
ax3 = Axis(fig[2, 1], xlabel = "k h / π", ylabel = "g_f",
           title = "prescribed filter 1 - sin^2n(θ/2)")
ax4 = Axis(fig[2, 2], xlabel = "tap index l", ylabel = "|c_l|", yscale = log10,
           title = "equivalent convolution kernel of a spline")
for (name, sym) in SCHEMES
    ls = startswith(name, "spline") ? :dash : (startswith(name, "LAG") ? :solid : :dot)
    lines!(ax1, th ./ pi, abs.(sym.(th)), label = name, linestyle = ls)
    lines!(ax2, th ./ pi, abs.(sym.(th)) .^ NSTEP, label = name, linestyle = ls)
end
for n in (4, 6, 8, 10, 12)
    lines!(ax3, th ./ pi, real.(fir_symbol.(Ref(flat_filter(n)), th)), label = "n = $n")
end
for p in (3, 5, 7)
    ls = [max(abs(C[(p, 0.3)][l+1]), 1e-18) for l = 0:32]
    lines!(ax4, 0:32, ls, label = "spline$p")
end
hlines!(ax4, [1e-8], color = :gray, linestyle = :dot)
ylims!(ax1, 0.9, 1.01)
ylims!(ax2, 0, 1.05)
ylims!(ax4, 1e-14, 2.0)
axislegend(ax2, position = :lb, framevisible = false)
axislegend(ax3, position = :lb, framevisible = false)
axislegend(ax4, position = :rt, framevisible = false)
out = abspath("interpolation_spectra.png")
save(out, fig)
println("\nfigure written to $out")
