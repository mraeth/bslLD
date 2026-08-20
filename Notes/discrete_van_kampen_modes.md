# Discrete van Kampen modes in BSL simulations

## Background

In a collisionless plasma, the linearised Vlasov equation admits a *continuum* of
solutions known as van Kampen modes.  For a given wavenumber $k$ the mode at velocity
$v$ oscillates at

$$
\omega = k v ,
$$

and the full continuum spans all $v \in (-\infty,\infty)$.  In the complex-$\omega$
plane these modes form a **branch cut** along the real axis rather than isolated poles.
They do not appear as roots of the dispersion relation $\det \mathbf{D}=0$; instead they
contribute to Landau damping of the discrete (pole) branches through the Landau
contour deformation.

In a semi-Lagrangian BSL simulation the velocity space is sampled on a finite grid of
$N_v$ points.  The continuum collapses to $N_v$ **discrete rays** visible in the
$\omega$–$k$ power spectrum.  Their properties differ between the parallel and
perpendicular propagation geometries.

---

## Parallel propagation ($k \parallel B_0$)

### Physics

The magnetic force $q\mathbf{v}\times\mathbf{B}$ has no component along $B_0$, so
parallel motion is unimpeded.  Particles at grid velocity $v_{\parallel,j}$ stream
freely along $x$, accumulating the phase factor

$$
e^{-i k_\parallel v_{\parallel,j} t}
$$

over the entire simulation.  These trajectories are never interrupted, so the beams are
fully coherent and produce sharp rays in the spectrum.

### Dispersion relation

$$
\boxed{\omega = k_\parallel \, v_{\parallel,j}}, \qquad j = 1,\ldots,N_v
$$

The rays fan out symmetrically around $\omega = 0$ with slopes equal to the parallel
velocity grid values.  The densest concentration occurs near $v \sim \pm v_{th}$ where
the Maxwellian weight is largest, creating the most visible rays.

### Scaling

Doubling $N_v$ doubles the number of rays, since each additional grid point introduces
one new beam frequency.  The recurrence time at which phase mixing reverses is

$$
t_\text{rec} = \frac{2\pi}{k_\parallel \, \Delta v_\parallel}, \qquad
\Delta v_\parallel = \frac{2 v_\text{max}}{N_v}.
$$

---

## Perpendicular propagation ($k \perp B_0$)

### Physics

The magnetic field confines perpendicular particle motion to gyro-orbits; there is no
net free streaming along $k_\perp$.  Physically, the resonance condition shifts from
the Landau form $\omega = kv$ to the cyclotron form $\omega = n\Omega_i$, and the
van Kampen continuum is replaced by the discrete Bernstein-mode spectrum between
harmonics.

In a BSL scheme with **operator splitting**, however, the $x$-advection substep

$$
x \leftarrow x - v_{x,j} \, \frac{\Delta t}{2}
$$

is executed with the *instantaneous* velocity $v_{x,j}$ — the current grid value —
before the velocity rotation (gyration) step is applied.  Each advection substep
therefore imprints the phase factor

$$
e^{-i k_\perp v_{x,j} \, \Delta t / 2}
$$

on every grid point $j$.  The subsequent gyration rotates $(v_x, v_y)$, which partially
cancels phase accumulation from successive $x$-advection steps over one cyclotron
period.  Because the rotated velocity lands between grid points and is interpolated, the
cancellation is imperfect and a residual discrete structure survives.

### Dispersion relation

$$
\boxed{\omega = k_\perp \, v_{x,j}}, \qquad j = 1,\ldots,N_{v_x}
$$

The form is identical to the parallel case, but the physical mechanism is a numerical
artefact of operator splitting rather than true free streaming.  Consequently the ray
**amplitudes are weaker** than in the parallel case (partial gyration cancellation), but
the number of rays and their slopes follow the same rule.

### Scaling

Doubling $N_{v_x}$ doubles the number of rays for the same reason as in the parallel
case: each discrete $v_{x,j}$ grid point contributes one beam frequency to the
$x$-advection substep.

---

## Comparison

| Property | Parallel | Perpendicular |
|---|---|---|
| Physical mechanism | Free streaming along $B_0$ | Operator-splitting artefact |
| Dispersion | $\omega = k_\parallel v_{\parallel,j}$ | $\omega = k_\perp v_{x,j}$ |
| Rays double with $N_v$? | Yes | Yes |
| Amplitude relative to physical modes | Strong | Weaker (gyration reduces it) |
| Present in continuous limit? | Branch cut (Landau continuum) | Absent (suppressed by $B$) |

---

## Suppression strategies

The rays live at **high velocity-space wavenumber** $\ell$ (fine-grained filaments in
$v$-space).  Operators that target high $\ell$ preferentially are the most effective.

| Method | Damping of mode $\ell$ | Selectivity |
|---|---|---|
| Krook / BGK | $\nu$ (flat) | None — damps physical modes equally |
| Lenard–Bernstein diffusion | $D\ell^2$ | Moderate |
| Velocity-space spectral filter | 0 for $\ell < \ell_c$, full for $\ell \geq \ell_c$ | Maximum |

A velocity-space low-pass filter (analogous to the `fourier_filter` already applied to
the fields in `step!`) is the most surgical choice: it leaves the low-$\ell$ physical
content (density, temperature, current) untouched while eliminating the high-$\ell$
recurrence filaments that generate the discrete rays.
