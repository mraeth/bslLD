# Drift-kinetic electron splitting: two candidate variants

The paper `paper_2025_hybrid_simulations.tex` presents two versions of the
Strang-splitting + leap-frog integrator for the drift-kinetic electron equation
(there are two `tab_splitting` tables in the source `.tex`, with the same
label). They differ in where the parallel advection of the leap-frog partner
`g` sits relative to the leap-frog kick. Only one of the two actually achieves
the promised O(Δt²) global convergence.

## Common setup

State per iteration: `(f^n, g^{n-1}, φ^n)`, with `g^{n-1}` a one-step-lagged
auxiliary distribution used as the "leg from behind" in the leap-frog. Common
sub-steps in every iteration:

- Solve `φ^n` from `f^n` (Poisson).
- z-shift by `Δt·v_z` (semi-Lagrangian, spectral, exact for these modes).
- v_z-shift by `-Δt·∂z φ` (uses the current φ).
- ExB kick `df = {f, φ/B_0}` (spectral Poisson bracket).
- Store some snapshot of `f^n` as the new `g^n`.

The disagreement is which of these steps is applied to `g` **before** the
leap-frog kick versus **after**.

## Variant A (first `tab_splitting` in the paper)

Row `n`, five sub-steps:

1. `g^{(1)} = g^{n-1}` v-shifted by `Δt` (using `φ^n`).
2. `g^{(2)} = g^{(1)}` z-shifted by `Δt`.
3. Kick: `f^{(1)} = g^{(2)} + 2Δt · {φ^n, f^n}`; `g^n = f^n`.
4. `f^{(2)} = f^{(1)}` z-shifted by `Δt`; solve `φ^{n+1}`.
5. `f^{n+1} = f^{(2)}` v-shifted by `Δt` (using `φ^{n+1}`).

The leap-frog kick reads from an already-advected `g^{(2)}`, and `g` is then
overwritten with the un-advected `f^n`.

## Variant B (second `tab_splitting` in the paper — Table-2 variant)

Row `n`:

- Solve `φ^n` from `f^n`.
- Kick: `f^{(1)} = g^{n-1} + 2Δt · {φ^n, f^n}` — the leap-frog uses the raw,
  un-advected `g^{n-1}` as it was stored at the end of the previous
  iteration.
- Build the new `g^n` from the current `f^n` by applying the same parallel
  shifts that will be applied to `f^{(1)}` in steps 4–5 (v-shift with `φ^n`,
  then z-shift).
- Step 4: `f^{(2)} = f^{(1)}` z-shifted by `Δt`; solve `φ^{n+1}`.
- Step 5: `f^{n+1} = f^{(2)}` v-shifted by `Δt` (using `φ^{n+1}`).

Equivalently: `g^n` is the (`z ∘ v`)-shifted `f^n` — a "one-step-ahead"
prediction of `f^{n+1}` computed with the parallel operators alone (no ExB).

The invariant preserved by Variant B is:

    at the start of every iteration,
        g^{n-1}  ≡  (z ∘ v)(f^{n-1}) using φ^{n-1}

so `g^{n-1}` is exactly the same operator applied to `f^{n-1}` that will be
applied to `f^n` in steps 4–5 of the previous iteration. This is the symmetry
Variant A breaks.

## Bootstrap

At `t = 0` only `f^0` is known. Variant B needs `g^0` in the invariant form
above. A one-step Lee-style bootstrap that produces both `f^1` and `g^0`:

```text
solve φ^0 from f^0
df = {f^0, φ^0}
g   = (z ∘ v)(f^0)                         # g^0
f  += Δt · df                               # explicit Euler kick, O(Δt²) local
f   = z(Δt) · f                             # step 4
solve φ^1
f   = v(Δt, φ^1) · f                        # step 5, → f^1
```

The local error introduced by the bootstrap is O(Δt²), i.e. the same order
that the whole scheme delivers globally, so it does not degrade the observed
rate.

## Why Variant A drops to first order

In the drift-kinetic problem the total advection experienced by `f` while
going from `t^{n−1}` to `t^{n+1}` comes from two places:

- Steps 1–2 in iteration `n` (applied to `g`, which then enters the kick).
- Steps 4–5 in iteration `n` (applied to `f` after the kick).

For a leap-frog to be second order over the 2Δt interval, the parallel-shift
combination effectively applied has to be symmetric around the midpoint `t^n`
(the point at which the kick uses `φ^n` and `f^n`). Variant A applies the
sequence

    v(Δt, φ^n) → z(Δt) → [kick] → z(Δt) → v(Δt, φ^{n+1})

Fusing the two z-shifts into `z(2Δt)` gives

    v(Δt, φ^n) · z(2Δt) · v(Δt, φ^{n+1})

Both v-shifts use fields evaluated at the two right-hand endpoints (`n` and
`n+1`) of the 2Δt interval `[n−1, n+1]`, not at the midpoint. Solving
`dv/dt = E(t)` over `[n−1, n+1]` with this quadrature is only first-order
accurate, and the whole scheme inherits that.

Variant B avoids this by ensuring `g^{n-1}` already carries one Δt worth of
parallel dynamics *evaluated using `φ^{n-1}`* — the field at the left endpoint
of the leap-frog window — so the effective quadrature becomes symmetric
around `t^n`.

## Numerical results

Driver: `bin/dk_electron_convergence.jl`. Grid `NX = 3, NV = 3`
(singleton perp velocity axes), 3×3D FFT-based spectral operators, periodic
BC, random-mode-mix initial data with `ε = 10^{-3}`, integration to `T = 1.0`,
reference `dt_ref = 10^{-3}`, sweep `dt ∈ {0.1, 0.05, 0.025, 0.0125}`. L₂
metric `sqrt(sum(abs2, f(dt) − f(dt_ref)))`.

### Variant A (Table 1)

| dt      | error   |
| :------ | :------ |
| 0.1     | 5.7·10⁻⁵ |
| 0.05    | 2.6·10⁻⁵ |
| 0.025   | 1.2·10⁻⁵ |
| 0.0125  | 5.6·10⁻⁶ |

Observed rates: `[1.14, 1.10, 1.10]` — first order.

### Variant B (Table 2, implemented)

| dt      | error   |
| :------ | :------ |
| 0.1     | 1.29·10⁻⁵ |
| 0.05    | 3.20·10⁻⁶ |
| 0.025   | 7.98·10⁻⁷ |
| 0.0125  | 1.99·10⁻⁷ |

Observed rates: `[2.01, 2.005, 2.008]` — clean second order.

Lee splitting (as a first-order sanity baseline, same driver): rates
`[1.02, 1.03, 1.06]` — first order as expected.

## Takeaway

Only Variant B delivers the O(Δt²) convergence claimed in the paper. The
subtle difference — whether the leap-frog kick sees `g` before or after its
parallel advection — moves the effective quadrature point for the
time-dependent `E(t)` from the endpoints of the 2Δt window to its centre, and
this is what buys the extra order. Variant A is what a literal reading of the
paper's first table produces, and it drops to first order in practice.
