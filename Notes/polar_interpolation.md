# Semi-Lagrangian interpolation on the polar velocity grid

This memo describes the radial interpolation used by the V-advection step in
[`src/kinetics/advectorPolar.jl`](../src/kinetics/advectorPolar.jl). It documents
two design choices that improve quality over plain cubic Lagrange while keeping
the kernel pointwise and GPU-friendly:

1. parity reflection across $r=0$ to handle the polar origin smoothly, and
2. Fritsch–Carlson PCHIP slope limiting along $r$ to suppress overshoot near maxima.

## Setting

The polar velocity grid has

$$
r_j = (j-\tfrac12)\,\Delta r,\qquad j=1,\dots,N_r,\qquad
\varphi_k = k\,\Delta\varphi,\qquad k=0,\dots,N_\varphi-1,
$$

i.e. the radial axis is cell-centered with $r_{\min}=\Delta r/2>0$; $r=0$ is
never a grid point. The V-advection step is split as

1. FFT along $\varphi$, producing complex coefficients $\hat f_m(\vec x, r_j)$
   for $m=0,1,\dots,N_\varphi-1$ in Julia FFT ordering;
2. for each grid point and FFT mode index $q$, evaluate the interpolated
   coefficient at the departure radius $r^{\mathrm{dep}}$ using a four-point
   radial stencil, then sum the Fourier series at the departure angle
   $\varphi^{\mathrm{dep}}$.

The radial step is the only non-spectral interpolation and is the focus of
this memo.

## Parity reflection at the pole

A smooth function $f(v_x,v_y)$ admits an angular Fourier expansion

$$
f(r,\varphi) = \sum_m \hat f_m(r)\, e^{i m\varphi}.
$$

The identity $(r,\varphi+\pi)\equiv(-r,\varphi)$ on Cartesian coordinates
forces the exact parity relation

$$
\hat f_m(-r) = (-1)^{|m|}\,\hat f_m(r).
$$

So for any radial stencil index $j\le 0$ we *define*

$$
\hat f_m(r_j) := (-1)^{|m|}\,\hat f_m(r_{1-j}),
$$

i.e. a "ghost" node at $r=-\Delta r/2$ inherits its value from the
innermost cell with sign $p_m = (-1)^{|m|}$. This is exact and adds no
new data.

With this rule the same four-point radial stencil applies uniformly down
to $r^{\mathrm{dep}}=0$ (and below); no zero-clamp, no asymmetric one-sided
stencil. In particular for $r^{\mathrm{dep}}\in[0,r_{\min})$ the stencil
becomes centered: $(-\Delta r/2,\;\Delta r/2,\;3\Delta r/2,\;5\Delta r/2)$.

The parity reflection also self-corrects symmetry drift: any non-physical
phi-asymmetry at the pole accumulated by previous steps is suppressed by
the explicit sign every time the origin region is sampled.

Implementation: see
[`_radial_sample`](../src/kinetics/advectorPolar.jl#L235-L241). The mode
parity $p_m$ is computed incrementally inside
[`_polar_v_fourier_poly_kernel!`](../src/kinetics/advectorPolar.jl) as the
loop walks through $m=0,1,2,\dots$ and the FFT-ordered negative tail; each
step flips the sign so no `pow` or `isodd` is evaluated per mode.

At the outer boundary ($r^{\mathrm{dep}}>r_{\max}$) the sampler returns
zero, which is appropriate for distributions that vanish in the high-energy
tail.

## Fritsch–Carlson PCHIP along $r$

The original kernel used cubic Lagrange interpolation on the four-point
stencil. Cubic Lagrange is fourth-order accurate in smooth regions but is
unbounded by the stencil values, so it overshoots at local extrema —
producing non-physical negative values for a positive $f$ near its peak
and ringing artifacts elsewhere.

We replace it with a Fritsch–Carlson piecewise cubic Hermite interpolant
(PCHIP). Given stencil values $c_0,c_1,c_2,c_3$ at equispaced nodes and a
local coordinate $t\in[0,1]$ in the central interval $[c_1,c_2]$, define
the one-sided slopes

$$
\delta_{01} = c_1-c_0,\qquad
\delta_{12} = c_2-c_1,\qquad
\delta_{23} = c_3-c_2.
$$

The limited slopes at the interior nodes are the weighted harmonic means

$$
s_1 =
\begin{cases}
0, & \delta_{01}\,\delta_{12}\le 0,\\[2pt]
\dfrac{3\,\delta_{01}\,\delta_{12}}{2\delta_{12}+\delta_{01}}, & \text{otherwise},
\end{cases}
\qquad
s_2 =
\begin{cases}
0, & \delta_{12}\,\delta_{23}\le 0,\\[2pt]
\dfrac{3\,\delta_{12}\,\delta_{23}}{\delta_{23}+2\delta_{12}}, & \text{otherwise}.
\end{cases}
$$

The interpolant is the standard cubic Hermite blend

$$
f(t) = h_{00}(t)\,c_1 + h_{10}(t)\,s_1 + h_{01}(t)\,c_2 + h_{11}(t)\,s_2,
$$

with

$$
h_{00}=2t^3-3t^2+1,\quad h_{10}=t^3-2t^2+t,\quad
h_{01}=-2t^3+3t^2,\quad h_{11}=t^3-t^2.
$$

Two properties of this scheme are essential here:

- **Maximum preservation.** At any local extremum the corresponding
  one-sided slope product is non-positive, so $s_j=0$ and the
  interpolant is bounded by $\min(c_1,c_2)\le f(t)\le \max(c_1,c_2)$.
  Maxima are preserved at grid resolution and undershoots into negative
  values are eliminated for unimodal data.
- **Smooth-region accuracy.** When the data is monotone and smooth, the
  harmonic-mean slopes coincide with second-order centered differences
  to leading order, and the scheme is $O(\Delta r^3)$ — one order below
  cubic Lagrange, but without overshoot.

The Fourier coefficients $c_q$ are complex. We apply PCHIP independently
to the real and imaginary parts. This preserves the boundedness property
on each component and avoids any branching on `abs(c)`, which is the
relevant property in practice: ringing in either component would feed
back into a phi-dependent oscillation of $f$ after the inverse Fourier
sum.

Implementation: see
[`_pchip_real`](../src/kinetics/advectorPolar.jl#L213-L230) and
[`_sample_radial_fourier_coeff`](../src/kinetics/advectorPolar.jl#L243-L274).
PCHIP cost is comparable to cubic Lagrange — a handful of extra multiplies
and two conditional zeros per stencil — and remains pointwise (no extra
global passes, GPU-friendly).

## Summary of the kernel

For each grid point $(\vec x_i, r_j, \varphi_k)$:

1. Compute the departure point in Cartesian velocity:
   $v_x = r_j\cos\varphi_k$, $v_y = r_j\sin\varphi_k$, then
   $(v_x,v_y) \mapsto (v_x-\delta_{\mu_x},\, v_y-\delta_{\mu_y})$
   with the $E$-field kick rotated into the polar frame.
2. Convert back to polar: $r^{\mathrm{dep}}, \varphi^{\mathrm{dep}}$.
3. For each Fourier mode index $q$:
   - Determine the parity $p_m=(-1)^{|m|}$ (incremental sign flips).
   - Sample $\hat f_m(r^{\mathrm{dep}})$ using the four-point radial
     stencil with PCHIP, applying $p_m$ to any ghost node at $j\le 0$.
   - Accumulate $\hat f_m(r^{\mathrm{dep}}) \, e^{i m\,(\varphi^{\mathrm{dep}}-\varphi_0)}$.
4. Write back the real part divided by $N_\varphi$.

This is non-conservative by design — the scheme is intended as the simple
and fast counterpart to a polygon remap.

## What this does *not* address

- **Mass conservation.** PCHIP does not preserve the integral of $f$;
  small mass drift will occur and remains the responsibility of the
  conservative remap path.
- **Outer-boundary smoothness.** The hard zero return for
  $r^{\mathrm{dep}}>r_{\max}$ is appropriate for thermal problems but
  is not smooth. If a problem populates the outer boundary significantly,
  consider a linear extrapolation through zero or enlarging $v_{\perp\max}$.
- **Phi resolution.** The angular interpolation is exact (Fourier
  evaluation), so $N_\varphi$ controls aliasing rather than smoothness.
