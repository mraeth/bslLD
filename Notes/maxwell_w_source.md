# Memo: Extending the Maxwell Solver to a Neutral Medium — Adding Electron Polarisation Drift

**Author:** [Your Name]
**Date:** 2026-05-18
**Subject:** Amendment to the neutral-medium Maxwell solver memo: inclusion of the electron polarisation drift current

---

## 1. Overview and Motivation

This memo amends the previous memo by adding one further physical effect to the electron current: the **electron polarisation drift**. Recall that the electron current was previously taken to be:

$$\mathbf{j}_e = n\,(\mathbf{E} \times \hat{\mathbf{z}})$$

This captures the $\mathbf{E}\times\mathbf{B}$ drift of the electron fluid. However, in a time-varying electric field, there is an additional contribution to the electron fluid velocity — the **polarisation drift** — arising from the inertia of the $\mathbf{E}\times\mathbf{B}$ motion itself. For electrons this is:

$$\mathbf{v}_\mathrm{pol} = -\frac{1}{\Omega_{e} B}\frac{d\mathbf{E}_\perp}{dt} = -\frac{m_e}{q_e B^2}\frac{\partial \mathbf{E}_\perp}{\partial t}$$

where $\Omega_e = q_e B / m_e$ is the electron gyrofrequency and $\mathbf{E}_\perp$ denotes the component of $\mathbf{E}$ perpendicular to $\hat{\mathbf{z}}$, i.e., the $xy$-plane projection $P_{xy}\mathbf{E}$.

The resulting additional current density is:

$$\mathbf{j}_\mathrm{pol} = -n\,\frac{m_e}{q_e B^2}\frac{\partial \mathbf{E}_\perp}{\partial t} = -\frac{n m_e}{q_e B^2}\,P_{xy}\frac{\partial \mathbf{E}}{\partial t}$$

where we use $P_{xy}$ to project onto the plane perpendicular to $\hat{\mathbf{z}}$.

The total electron current becomes:

$$\mathbf{j}_e = n(\mathbf{E}\times\hat{\mathbf{z}}) - \frac{n m_e}{q_e B^2}\,P_{xy}\frac{\partial \mathbf{E}}{\partial t}$$

and the total current:

$$\mathbf{j} = \mathbf{j}_i - \mathbf{j}_e = \mathbf{j}_i - n(\mathbf{E}\times\hat{\mathbf{z}}) + \frac{n m_e}{q_e B^2}\,P_{xy}\frac{\partial \mathbf{E}}{\partial t}$$

The key structural consequence is that $\mathbf{j}_\mathrm{pol}$ contains $\partial_t\mathbf{E}$, which means it **modifies the coefficient of the displacement current** in Ampère's law. This changes the wave speeds and, more importantly, changes the operator that must be inverted at each time step.

---

## 2. Modified Ampère's Law

Substituting the full electron current into Ampère's law:

$$\nabla\times\mathbf{B} = \mu_0\mathbf{j}_i - \mu_0 n(\mathbf{E}\times\hat{\mathbf{z}}) + \frac{\mu_0 n m_e}{q_e B^2}\,P_{xy}\frac{\partial\mathbf{E}}{\partial t} + \frac{1}{c^2}\frac{\partial\mathbf{E}}{\partial t}$$

Collecting the $\partial_t\mathbf{E}$ terms on the right:

$$\nabla\times\mathbf{B} = \mu_0\mathbf{j}_i - \mu_0 n(\mathbf{E}\times\hat{\mathbf{z}}) + \left(\frac{1}{c^2}I + \frac{\mu_0 n m_e}{q_e B^2}P_{xy}\right)\frac{\partial\mathbf{E}}{\partial t}$$

Defining the **modified permittivity tensor**:

$$\boldsymbol{\varepsilon} := \varepsilon_0\left(I + \frac{n m_e c^2}{q_e B^2}P_{xy}\right) = \varepsilon_0\left(I + \chi\, P_{xy}\right)$$

where the dimensionless susceptibility is:

$$\chi := \frac{n m_e c^2}{q_e B^2} = \frac{\omega_{pe}^2}{\Omega_e^2} \cdot \frac{c^2}{c^2} = \frac{n m_e}{\varepsilon_0 B^2} \cdot \varepsilon_0 c^2 \cdot \frac{1}{c^2}\frac{m_e}{m_e} = \frac{n m_e c^2}{q_e B^2}$$

the modified Ampère law becomes:

$$\boldsymbol{\varepsilon}\,\frac{\partial\mathbf{E}}{\partial t} = \frac{1}{\mu_0}\nabla\times\mathbf{B} - \mathbf{j}_i + n(\mathbf{E}\times\hat{\mathbf{z}})$$

Solving for $\partial_t\mathbf{E}$ requires **inverting the tensor $\boldsymbol{\varepsilon}$**. Since $\boldsymbol{\varepsilon} = \varepsilon_0(I + \chi P_{xy})$ and $P_{xy}^2 = P_{xy}$, the inverse is:

$$\boldsymbol{\varepsilon}^{-1} = \frac{1}{\varepsilon_0}\left(I - \frac{\chi}{1+\chi}P_{xy}\right) = \frac{1}{\varepsilon_0}\left(P_z + \frac{1}{1+\chi}P_{xy}\right)$$

where $P_z = I - P_{xy} = \mathrm{diag}(0,0,1)$ is the projection onto $\hat{\mathbf{z}}$. The evolution equation for $\mathbf{E}$ is therefore:

$$\frac{\partial\mathbf{E}}{\partial t} = \boldsymbol{\varepsilon}^{-1}\left(\frac{1}{\mu_0}\nabla\times\mathbf{B} - \mathbf{j}_i + n(\mathbf{E}\times\hat{\mathbf{z}})\right)$$

Written component by component:

$$\frac{\partial E_x}{\partial t} = \frac{1}{\varepsilon_0(1+\chi)}\left(\frac{1}{\mu_0}(\nabla\times\mathbf{B})_x - j_{ix} + n E_y\right)$$

$$\frac{\partial E_y}{\partial t} = \frac{1}{\varepsilon_0(1+\chi)}\left(\frac{1}{\mu_0}(\nabla\times\mathbf{B})_y - j_{iy} - n E_x\right)$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\varepsilon_0}\left(\frac{1}{\mu_0}(\nabla\times\mathbf{B})_z - j_{iz}\right)$$

The $z$-component is unchanged from the previous memo. The $x$ and $y$ components acquire a **reduced effective speed of light** in the perpendicular plane:

$$\tilde{c}_\perp^2 = \frac{c^2}{1+\chi}$$

For $\chi \gg 1$ (dense plasma, weak field), $\tilde{c}_\perp \ll c$ — the perpendicular electromagnetic waves are strongly slowed by the electron inertia loading. This is the physical content of the polarisation drift.

---

## 3. Structural Change to the Operator $L$

The full system is still of the form $\partial_t u = L u + S$, but the operator $L$ now reads:

$$\frac{d}{dt}\begin{pmatrix}\mathbf{E}\\ \mathbf{B}\end{pmatrix} = \begin{pmatrix} \boldsymbol{\varepsilon}^{-1}\left(\frac{n}{\mu_0 c^2}M_\times\right) & \boldsymbol{\varepsilon}^{-1}\frac{1}{\mu_0}\nabla\times \\ -\nabla\times & 0 \end{pmatrix}\begin{pmatrix}\mathbf{E}\\ \mathbf{B}\end{pmatrix} + \begin{pmatrix} -\boldsymbol{\varepsilon}^{-1}\mathbf{j}_i \\ \mathbf{0} \end{pmatrix}$$

The key observation is that $\boldsymbol{\varepsilon}^{-1}$ is **diagonal** and **spatially varying** (through $\chi = \chi(\mathbf{x})$) but acts differently on the $xy$-components and the $z$-component. It does not destroy the block structure — it simply rescales the $E_x$ and $E_y$ evolution equations by a factor $1/(1+\chi(\mathbf{x}))$.

### 3.1 Is $L$ Still Skew-Hermitian?

Define the weighted inner product:

$$\langle u, u\rangle_\varepsilon := \int \left(\mathbf{E}\cdot\boldsymbol{\varepsilon}\mathbf{E} + \frac{1}{\mu_0}|\mathbf{B}|^2\right)d^3x$$

This is the physical electromagnetic energy including the polarisation contribution. With respect to this inner product, the operator $L$ is **skew-symmetric** — meaning the system conserves $\langle u, u\rangle_\varepsilon$ in the absence of sources. The Crank–Nicolson discretisation with respect to this weighted norm remains stable and energy-conserving.

However, with respect to the **standard** (unweighted) inner product used in the previous implementation, $L$ is no longer skew-Hermitian because $\boldsymbol{\varepsilon}^{-1}$ is not a scalar multiple of the identity. This has consequences for the Strang splitting, discussed below.

---

## 4. Impact on the Operator Splitting

### 4.1 Splitting Substep A: Modified Drift Update

Substep A now advances:

$$\frac{\partial \mathbf{E}}{\partial t} = \boldsymbol{\varepsilon}^{-1}\left(\frac{n}{\mu_0 c^2}M_\times\,\mathbf{E} - \mathbf{j}_i\right), \qquad \frac{\partial\mathbf{B}}{\partial t} = 0$$

Using $\boldsymbol{\varepsilon}^{-1} = \varepsilon_0^{-1}(P_z + (1+\chi)^{-1}P_{xy})$, this becomes:

- For the $x$ and $y$ components: $\partial_t E_{x,y} = \frac{\omega_{x,y}}{1+\chi} (M_\times \mathbf{E})_{x,y} - \frac{1}{\varepsilon_0(1+\chi)}j_{i,x,y}$
- For the $z$ component: $\partial_t E_z = -\frac{1}{\varepsilon_0}j_{iz}$ (no drift coupling)

where $\omega = n/\varepsilon_0$ as before. The rotation angle per unit time in the $xy$-plane is now **reduced** to:

$$\tilde{\omega} = \frac{\omega}{1+\chi} = \frac{n/\varepsilon_0}{1 + nm_e c^2/(q_e B^2)}$$

The CN update for the $xy$-block of Substep A carries through exactly as before (Section 5.4 of the previous memo) with the replacement $\omega \to \tilde{\omega}$ and $1/\varepsilon_0 \to 1/(\varepsilon_0(1+\chi))$ in the forcing term. The closed-form Cayley inverse still applies:

$$\tilde{\beta} = \frac{\tau\tilde{\omega}}{2}, \qquad \tilde{\mathrm{denom}} = \frac{1}{1+\tilde{\beta}^2}$$

```julia
function step_drift_pol!(E::VectorField, ji::VectorField,
                         n::ScalarField, B_magnitude::ScalarField,
                         grid::Grid, dt::Real, eps0::Real,
                         me::Real, qe::Real, c::Real)
    tau   = dt / 2
    chi   = (n.data .* me .* c^2) ./ (qe .* B_magnitude.data .^ 2)
    omega = n.data ./ eps0
    omega_tilde = omega ./ (1 .+ chi)             # reduced drift frequency
    beta  = tau .* omega_tilde ./ 2
    denom = 1 ./ (1 .+ beta .^ 2)

    Ex, Ey, Ez = E[1].data, E[2].data, E[3].data
    jx, jy, jz = ji[1].data, ji[2].data, ji[3].data

    # Effective forcing: reduced by (1+χ) in xy-plane
    force_scale = tau ./ (eps0 .* (1 .+ chi))

    rhs_x = Ex .+ beta .* Ey .- force_scale .* jx
    rhs_y = Ey .- beta .* Ex .- force_scale .* jy

    E[1].data .= denom .* (rhs_x .+ beta .* rhs_y)
    E[2].data .= denom .* (rhs_y .- beta .* rhs_x)
    E[3].data .= Ez .- (tau / eps0) .* jz          # z unchanged by drift
end
```

### 4.2 Splitting Substep B: Modified Curl Step

The vacuum Crank–Nicolson substep B previously used:

$$\frac{\partial \mathbf{E}}{\partial t} = c^2\nabla\times\mathbf{B}$$

With the polarisation drift, this becomes:

$$\frac{\partial E_{x,y}}{\partial t} = \frac{c^2}{1+\chi}\,(\nabla\times\mathbf{B})_{x,y}, \qquad \frac{\partial E_z}{\partial t} = c^2\,(\nabla\times\mathbf{B})_z$$

Since $\chi = \chi(\mathbf{x})$ is spatially varying, the **Fourier-space curl operator is no longer diagonal** when combined with $\boldsymbol{\varepsilon}^{-1}$: the factor $(1+\chi(\mathbf{x}))^{-1}$ couples Fourier modes together. The original closed-form Cayley update of Substep B is therefore **no longer directly applicable**.

Two strategies are available:

---

## 5. Strategy 1 — Uniform $\chi$ Approximation

If $n(\mathbf{x})$ and $B(\mathbf{x})$ are nearly spatially uniform, $\chi \approx \bar{\chi}$ (a global constant), and the modified Ampère law becomes:

$$\frac{\partial \mathbf{E}}{\partial t} \approx \frac{c^2}{1+\bar{\chi}}\,\nabla\times\mathbf{B} + \ldots$$

In this case, the Fourier-space Cayley analysis of the companion memo goes through unchanged, with the substitution:

$$c^2 \;\longrightarrow\; \tilde{c}^2 = \frac{c^2}{1+\bar{\chi}} \quad \text{for the } E_x, E_y \text{ components}$$

The closed-form update in Substep B becomes:

$$\hat{E}_{x,y}^{n+1} = \frac{1}{1+\tilde{\alpha}^2}\left[(1-\tilde{\alpha}^2)\hat{E}_{x,y}^{n} + \tilde{c}^2\Delta t\,(\hat{\nabla\times\mathbf{B}})_{x,y}^{n}\right]$$

$$\hat{E}_z^{n+1} = \frac{1}{1+\alpha_z^2}\left[(1-\alpha_z^2)\hat{E}_z^{n} + c^2\Delta t\,(\hat{\nabla\times\mathbf{B}})_z^{n}\right]$$

where $\tilde{\alpha} = \tilde{c}|\mathbf{k}|\Delta t/2$ and $\alpha_z = c|\mathbf{k}|\Delta t/2$. The $\mathbf{B}$ update is unchanged. This is the recommended approach when spatial variation of $\chi$ is weak.

The implementation change in `step_cn!` is minimal — pass a modified speed `c_tilde` for the $xy$ components:

```julia
function step_cn_pol!(E::VectorField, B::VectorField,
                      grid::Grid, dt::Real, c::Real, chi_bar::Real)
    c_perp = c / sqrt(1 + chi_bar)    # reduced perpendicular phase speed

    curlE = curl(E, grid)
    curlB = curl(B, grid)

    # Per-mode α² values differ for xy and z
    # alpha2_perp(k) = (c_perp * |k| * dt/2)²
    # alpha2_z(k)    = (c     * |k| * dt/2)²
    # ... spectral loop with component-dependent prefactor ...

    for d in 1:2    # x and y: use c_perp
        apply_cayley!(E[d], B[d], curlB[d], curlE[d], grid, dt, c_perp)
    end
    apply_cayley!(E[3], B[3], curlB[3], curlE[3], grid, dt, c)    # z: use c
end
```

---

## 6. Strategy 2 — Spatially Varying $\chi$: Modified Implicit Step

When $\chi(\mathbf{x})$ varies significantly in space, the product $\chi(\mathbf{x})\cdot(\nabla\times\mathbf{B})$ mixes Fourier modes and must be handled in physical space. The CN update for Substep B becomes:

$$\frac{\mathbf{E}^{n+1} - \mathbf{E}^n}{\Delta t} = \boldsymbol{\varepsilon}^{-1}\frac{1}{\mu_0}\nabla\times\left(\frac{\mathbf{B}^{n+1}+\mathbf{B}^n}{2}\right)$$

$$\frac{\mathbf{B}^{n+1} - \mathbf{B}^n}{\Delta t} = -\nabla\times\left(\frac{\mathbf{E}^{n+1}+\mathbf{E}^n}{2}\right)$$

Substituting the second into the first to eliminate $\mathbf{B}^{n+1}$:

$$\mathbf{E}^{n+1} - \mathbf{E}^n = \frac{\Delta t}{\mu_0}\boldsymbol{\varepsilon}^{-1}\nabla\times\left(\mathbf{B}^n - \frac{\Delta t}{2}\nabla\times\frac{\mathbf{E}^{n+1}+\mathbf{E}^n}{2}\right)$$

Rearranging:

$$\left(I + \frac{\Delta t^2}{4}\boldsymbol{\varepsilon}^{-1}\nabla\times\nabla\times\right)\mathbf{E}^{n+1} = \mathbf{E}^n + \frac{\Delta t}{\mu_0}\boldsymbol{\varepsilon}^{-1}\nabla\times\mathbf{B}^n - \frac{\Delta t^2}{4}\boldsymbol{\varepsilon}^{-1}\nabla\times\nabla\times\mathbf{E}^n$$

The right-hand side is known at time $t^n$ and can be computed explicitly. The left-hand side involves the operator:

$$\mathcal{L} := I + \frac{\Delta t^2}{4}\boldsymbol{\varepsilon}^{-1}\nabla\times\nabla\times$$

For the $xy$-components this is $I + \frac{\tilde{c}^2(\mathbf{x})\Delta t^2}{4}\nabla\times\nabla\times$ with a spatially varying coefficient — a **variable-coefficient Helmholtz-type operator**. This no longer has a closed-form inverse.

### 6.1 Recommended Iterative Solver

The operator $\mathcal{L}$ is **symmetric positive definite** (SPD) with respect to the $\boldsymbol{\varepsilon}$-weighted inner product, which makes it amenable to a **preconditioned conjugate gradient (PCG)** solve. The natural preconditioner is the constant-coefficient version of $\mathcal{L}$ with $\chi \to \bar{\chi}$, which is diagonal in Fourier space and therefore cheap to invert spectrally:

$$\mathcal{L}_0 := I + \frac{\bar{\tilde{c}}^2\Delta t^2}{4}\nabla\times\nabla\times$$

$$\hat{\mathcal{L}}_0^{-1}(\mathbf{k}) = \frac{1}{1 + \bar{\tilde{c}}^2 |\mathbf{k}|^2 \Delta t^2 / 4}$$

This preconditioner is applied in Fourier space (one FFT pair per PCG iteration). Convergence is fast when $\chi$ does not vary too strongly, typically requiring $\mathcal{O}(5\text{–}10)$ iterations.

```julia
function step_cn_pol_variable!(E, B, grid, dt, c, chi_field, eps0, mu0;
                                tol=1e-10, maxiter=20)
    # Compute explicit right-hand side
    curlB = curl(B, grid)
    curlE = curl(E, grid)
    chi   = chi_field.data
    eps_inv_perp = 1 ./ (eps0 .* (1 .+ chi))   # spatially varying

    rhs = VectorField([
        ScalarField(E[1].data .+ dt .* eps_inv_perp .* curlB[1].data ./ mu0
                              .- (dt^2/4) .* eps_inv_perp .* (curl(curlE,grid))[1].data),
        ScalarField(E[2].data .+ dt .* eps_inv_perp .* curlB[2].data ./ mu0
                              .- (dt^2/4) .* eps_inv_perp .* (curl(curlE,grid))[2].data),
        ScalarField(E[3].data .+ dt ./ eps0         .* curlB[3].data ./ mu0
                              .- (dt^2/4) ./ eps0   .* (curl(curlE,grid))[3].data)
    ])

    # PCG solve: (I + Δt²/4 ε⁻¹ ∇×∇×) E^{n+1} = rhs
    chi_bar   = mean(chi)
    c_bar     = c / sqrt(1 + chi_bar)
    precond   = k -> 1 / (1 + c_bar^2 * sum(k.^2) * dt^2 / 4)   # spectral precond

    E_new = pcg_solve(E -> apply_L(E, chi, grid, dt, c, eps0),
                      rhs, precond; tol=tol, maxiter=maxiter)

    # Update B via Faraday using the averaged E
    curlE_avg = curl((E + E_new) / 2, grid)
    for d in 1:3
        B[d].data .-= dt .* curlE_avg[d].data
    end
    copyto!.(getfield.(E.data, :data), getfield.(E_new.data, :data))
end
```

### 6.2 When is the Iterative Solve Necessary?

| Condition | Recommended approach |
|-----------|---------------------|
| $\chi \ll 1$ everywhere | Polarisation drift negligible; use original solver |
| $\chi \approx \bar{\chi}$ (weakly varying) | Strategy 1: uniform $\bar{\chi}$, closed-form Cayley |
| $\chi$ strongly varying, $\delta\chi/\bar{\chi} \lesssim 1$ | Strategy 1 as predictor + one correction step |
| $\chi$ strongly varying, $\delta\chi/\bar{\chi} \sim 1$ | Strategy 2: PCG with spectral preconditioner |

---

## 7. Impact on the Magnetostatic Limit

In the magnetostatic limit ($c\to\infty$, displacement current dropped), the polarisation drift term survives because it involves $\partial_t\mathbf{E}$, which remains finite even as $c\to\infty$ (the polarisation drift velocity is $\sim \partial_t\mathbf{E}/\Omega_e B$, which is a physical velocity independent of $c$). The modified Ampère law in this limit is:

$$\nabla\times\mathbf{B} = \mu_0\mathbf{j}_i - \mu_0 n(\mathbf{E}\times\hat{\mathbf{z}}) + \frac{\mu_0 n m_e}{q_e B^2}\,P_{xy}\frac{\partial\mathbf{E}}{\partial t}$$

Rearranging for $\partial_t\mathbf{E}$:

$$\frac{\mu_0 n m_e}{q_e B^2}\,P_{xy}\frac{\partial\mathbf{E}}{\partial t} = \nabla\times\mathbf{B} - \mu_0\mathbf{j}_i + \mu_0 n(\mathbf{E}\times\hat{\mathbf{z}})$$

This means $\partial_t\mathbf{E}$ (in the $xy$-plane) is **still determined algebraically** from the current balance — but now it appears on both sides of the equation, requiring the inversion of $P_{xy}$. Since $n > 0$ this is trivial:

$$\frac{\partial E_{x,y}}{\partial t} = \frac{q_e B^2}{\mu_0 n m_e}\left(\nabla\times\mathbf{B} - \mu_0\mathbf{j}_i + \mu_0 n(\mathbf{E}\times\hat{\mathbf{z}})\right)_{x,y}$$

This can be interpreted as an **equation of motion for the polarisation drift velocity** rather than a wave equation. The $z$-component of $\mathbf{E}$ still requires the auxiliary Ohm's law condition as before.

Crucially, in the magnetostatic limit the polarisation drift introduces a **new dynamical timescale** — the drift-wave frequency $\omega_\mathrm{dw} \sim k_\perp v_E$ — which can constrain $\Delta t$ even without light waves. This timescale is typically much longer than $\Omega_e^{-1}$ but shorter than the Alfvén crossing time, and must be resolved.

---

## 8. Impact on the Vlasov-Coupled System

In the Vlasov-coupled case, $n$ and $\mathbf{j}_i$ are ion moments. The electron polarisation drift introduces an additional self-consistency requirement: the value of $\chi(\mathbf{x},t) = nm_ec^2/(q_eB^2)$ now depends on the **instantaneous ion density** $n = \int f_i\,d^3v$, which changes every step.

The consequences are:

1. **$\chi$ must be recomputed from moments every step**, just as $n$ and $\mathbf{j}_i$ are. This is a cheap pointwise operation given $n$ and $B$.

2. **The field update substep (Step 4 in the Vlasov loop)** must use `step_drift_pol!` and either `step_cn_pol!` (uniform $\chi$) or `step_cn_pol_variable!` (varying $\chi$) instead of their non-polarisation counterparts.

3. **The Boris push** for the Vlasov characteristics is unchanged — it uses the total $\mathbf{E}$ and $\mathbf{B}$ fields, and the polarisation drift effect is already encoded in those fields by the modified Maxwell solver.

4. **Energy accounting** requires updating the energy norm: the electromagnetic energy now includes the polarisation energy:

$$\mathcal{H}_\mathrm{EM} = \frac{1}{2}\int\left(\mathbf{E}\cdot\boldsymbol{\varepsilon}\mathbf{E} + \frac{|\mathbf{B}|^2}{\mu_0}\right)d^3x = \frac{1}{2}\int\left(\varepsilon_0(1+\chi)(E_x^2+E_y^2) + \varepsilon_0 E_z^2 + \frac{|\mathbf{B}|^2}{\mu_0}\right)d^3x$$

The total conserved energy is $\mathcal{H}_\mathrm{EM} + \mathcal{H}_\mathrm{kin}$ where $\mathcal{H}_\mathrm{kin} = \frac{m_i}{2}\int v^2 f_i\,d^6z$.

---

## 9. Summary of All Changes

The following table summarises the cumulative impact of adding the electron polarisation drift relative to all prior versions of the solver:

| Aspect | Without pol. drift | With pol. drift |
|--------|--------------------|-----------------|
| Electron current $\mathbf{j}_e$ | $n(\mathbf{E}\times\hat{\mathbf{z}})$ | $n(\mathbf{E}\times\hat{\mathbf{z}}) - \frac{nm_e}{q_eB^2}P_{xy}\partial_t\mathbf{E}$ |
| Effective permittivity | $\varepsilon_0 I$ | $\varepsilon_0(I + \chi P_{xy})$ |
| Perpendicular wave speed | $c$ | $\tilde{c} = c/\sqrt{1+\chi}$ |
| Drift rotation frequency | $\omega = n/\varepsilon_0$ | $\tilde{\omega} = n/(\varepsilon_0(1+\chi))$ |
| Substep A: `step_drift!` | $\omega, 1/\varepsilon_0$ | $\tilde{\omega}, 1/(\varepsilon_0(1+\chi))$ |
| Substep B: `step_cn!` | Closed-form Cayley, speed $c$ | Closed-form with $\tilde{c}$ (uniform $\chi$) or PCG (variable $\chi$) |
| Magnetostatic $\partial_t\mathbf{E}$ | Not present | Algebraic from modified Ampère |
| Vlasov coupling | $\chi$ not needed | $\chi$ computed from moments each step |
| Energy norm | $\varepsilon_0|\mathbf{E}|^2 + |\mathbf{B}|^2/\mu_0$ | $\mathbf{E}\cdot\boldsymbol{\varepsilon}\mathbf{E} + |\mathbf{B}|^2/\mu_0$ |
| New parameters required | — | $m_e$, $q_e$, $B$ magnitude field |

---

## 10. Notes and TODOs

- [ ] Implement `step_drift_pol!` with spatially varying `chi` field
- [ ] Implement `step_cn_pol!` for the uniform-$\chi$ approximation
- [ ] Implement `apply_L` operator and `pcg_solve` for the variable-$\chi$ case
- [ ] Add `chi` diagnostic output: monitor $\max\chi$ and $\delta\chi/\bar{\chi}$ to decide which strategy is needed
- [ ] Update energy diagnostic to use the $\boldsymbol{\varepsilon}$-weighted norm
- [ ] Verify that `chi` computed from Vlasov moments is consistent with the $B$ field magnitude used in $\boldsymbol{\varepsilon}$
- [ ] Check reduced wave speed $\tilde{c}$ to confirm CFL is not tightened unexpectedly (it should be relaxed since $\tilde{c} < c$)