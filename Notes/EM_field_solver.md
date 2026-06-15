### Particle Equations

Use ion normalization
$$
\Omega_i=\frac{eB_0}{m_i},
\qquad
v_{\mathrm{th},i}=\sqrt{\frac{T_i}{m_i}},
\qquad
\rho_i=\frac{v_{\mathrm{th},i}}{\Omega_i}.
$$

Dimensionless variables are defined by
$$
t=\Omega_i t_{\rm phys},
\qquad
\mathbf{x}=\frac{\mathbf{x}_{\rm phys}}{\rho_i},
\qquad
\mathbf{v}=\frac{\mathbf{v}_{\rm phys}}{v_{\mathrm{th},i}},
$$

$$
\mathbf{E}=\frac{\mathbf{E}_{\rm phys}}{v_{\mathrm{th},i}B_0},
\qquad
\mathbf{B}=\frac{\mathbf{B}_{\rm phys}}{B_0}.
$$

From here on, all quantities are dimensionless unless stated otherwise.

The ion Vlasov equation is
$$
\partial_t f_i
+
\mathbf v\cdot\nabla f_i
+
\left(
\mathbf E+\mathbf v\times \mathbf B
\right)\cdot\nabla_{\mathbf v} f_i
=0.
$$

For a uniform guide field,
$$
\mathbf B=\mathbf e_z.
$$

The electron drift-kinetic equation is
$$
\partial_t f_e
+
\mathbf u_{E\times B}\cdot\nabla f_e
+
v_z\,\partial_z f_e
-
\frac{m_i}{m_e}E_\parallel\,\partial_{v_z}f_e
=0,
$$
where
$$
E_\parallel=\mathbf b\cdot \mathbf E,
\qquad
\mathbf b=\frac{\mathbf B}{B},
$$
and
$$
\mathbf u_{E\times B}
=
\frac{\mathbf E\times \mathbf B}{B^2}.
$$

For the guide-field approximation $\mathbf B=\mathbf e_z$, this becomes
$$
\mathbf u_{E\times B}
=
\mathbf E\times \mathbf e_z,
\qquad
E_\parallel=E_z.
$$

The coefficient in the electron parallel acceleration is $m_i/m_e$ because velocities are normalized to the ion thermal speed.

---

### Current Decomposition

The displacement current is neglected:
$$
\epsilon_0\frac{\partial \mathbf E}{\partial t}\to 0.
$$

The total current is
$$
\mathbf J=\mathbf J_i+\mathbf J_e.
$$

The electron current is decomposed as
$$
\mathbf J_e
=
\mathbf J_{e,E\times B}
+
\mathbf J_{e,p}
+
\mathbf J_{e,\parallel}.
$$

In normalized variables,
$$
\mathbf J_{e,E\times B}
=
-n_e
\frac{\mathbf E\times \mathbf B}{B^2},
$$

$$
\mathbf J_{e,p}
=
n_e
\frac{m_e}{m_i}
\frac{1}{B^2}
\frac{D\mathbf E_\perp}{Dt},
$$

$$
\mathbf J_{e,\parallel}
=
-n_e u_{e,\parallel}\mathbf b.
$$

Here
$$
\frac{D}{Dt}
=
\partial_t+\mathbf u_{E\times B}\cdot\nabla
$$
unless a different convective derivative is specified.

The term $\mathbf J_{e,p}$ is the electron polarization current, not the vacuum displacement current.

For the guide-field approximation $\mathbf B=\mathbf e_z$, one has $B=1$ and $\mathbf b=\mathbf e_z$, so
$$
\mathbf J_{e,E\times B}
=
-n_e \mathbf E\times \mathbf e_z,
$$

$$
\mathbf J_{e,p}
=
n_e
\frac{m_e}{m_i}
\frac{D\mathbf E_\perp}{Dt},
$$

$$
\mathbf J_{e,\parallel}
=
-n_e u_{e,\parallel}\mathbf e_z.
$$

---

### Normalized Ampère's Law

Use the current and density normalizations
$$
\mathbf J=\frac{\mathbf J_{\rm phys}}{en_0v_{\mathrm{th},i}},
\qquad
n=\frac{n_{\rm phys}}{n_0}.
$$

Define the ion plasma beta
$$
\beta_i
=
\frac{2\mu_0 n_0 T_i}{B_0^2}.
$$

Neglecting the displacement current, Ampère's law becomes
$$
\nabla\times\mathbf B
=
\frac{\beta_i}{2}\mathbf J.
$$

Using the electron-current decomposition,
$$
\nabla\times\mathbf B
=
\frac{\beta_i}{2}
\left[
\mathbf J_i
-
n_e
\frac{\mathbf E\times\mathbf B}{B^2}
+
n_e
\frac{m_e}{m_i}
\frac{1}{B^2}
\frac{D\mathbf E_\perp}{Dt}
-
n_e u_{e,\parallel}\mathbf b
\right].
$$

For the guide-field approximation $\mathbf B=\mathbf e_z$, this reduces to
$$
\nabla\times\mathbf B
=
\frac{\beta_i}{2}
\left[
\mathbf J_i
-
n_e \mathbf E\times\mathbf e_z
+
n_e
\frac{m_e}{m_i}
\frac{D\mathbf E_\perp}{Dt}
-
n_e u_{e,\parallel}\mathbf e_z
\right].
$$

---

### Parallel and Perpendicular Components

The parallel component of Ampère's law is
$$
\mathbf b\cdot\left(\nabla\times\mathbf B\right)
=
\frac{\beta_i}{2}
\left(
J_{i,\parallel}
-
n_e u_{e,\parallel}
\right).
$$

Equivalently, as a vector equation,
$$
\left(\nabla\times\mathbf B\right)_\parallel
=
\frac{\beta_i}{2} \left( J_{i,\parallel}
-
n_e u_{e,\parallel}
\right)\mathbf b.
$$

The perpendicular component is
$$
\left(\nabla\times\mathbf B\right)_\perp
=
\frac{\beta_i}{2}
\left[
\mathbf J_{i,\perp}
-
n_e
\frac{\mathbf E\times\mathbf B}{B^2}
+
n_e
\frac{m_e}{m_i}
\frac{1}{B^2}
\frac{D\mathbf E_\perp}{Dt}
\right].
$$

For $\mathbf B=\mathbf e_z$, this becomes
$$
\left(\nabla\times\mathbf B\right)_\perp
=
\frac{\beta_i}{2}
\left[
\mathbf J_{i,\perp}
-
n_e\mathbf E\times\mathbf e_z
+
n_e
\frac{m_e}{m_i}
\frac{D\mathbf E_\perp}{Dt}
\right].
$$

---

### Parallel Electric Field from Faraday's Law

Faraday's law is
$$
\partial_t \mathbf B
=
-\nabla\times\mathbf E.
$$

Taking the curl gives
$$
\partial_t\left(\nabla\times\mathbf B\right)
=
-\nabla\times\nabla\times\mathbf E.
$$

Using normalized Ampère's law,
$$
\nabla\times\mathbf B
=
\frac{\beta_i}{2}\mathbf J,
$$
one obtains
$$
\frac{\beta_i}{2}\partial_t\mathbf J
=
-\nabla\times\nabla\times\mathbf E.
$$

The parallel component is therefore
$$
\frac{\beta_i}{2}
\partial_t
\left(
J_{i,\parallel}
-
J_{e,\parallel}
\right)
=
-\mathbf b\cdot
\left(
\nabla\times\nabla\times\mathbf E
\right).
$$

For a uniform guide field $\mathbf b=\mathbf e_z$, write
$$
\mathbf E
=
\mathbf E_\perp
+
E_\parallel \mathbf e_z.
$$

Then
$$
\mathbf e_z\cdot
\left(
\nabla\times\nabla\times\mathbf E
\right)
=
\partial_\parallel
\left(
\nabla_\perp\cdot\mathbf E_\perp
\right)
-
\nabla_\perp^2 E_\parallel.
$$

Hence the parallel electric field satisfies
$$
\nabla_\perp^2 E_\parallel
-
\partial_\parallel
\left(
\nabla_\perp\cdot\mathbf E_\perp
\right)
=
\frac{\beta_i}{2}
\partial_t
\left(
J_{i,\parallel}
-
J_{e,\parallel}
\right).
$$

Equivalently,
$$
\nabla_\perp^2 E_\parallel
=
\frac{\beta_i}{2}
\partial_t
\left(
J_{i,\parallel}
-
J_{e,\parallel}
\right)
+
\partial_\parallel
\left(
\nabla_\perp\cdot\mathbf E_\perp
\right).
$$
Using only the background magnetic field
$$
\mathbf B_0=\hat{\mathbf z},
$$
the parallel direction is $ \parallel = z $. Hence
$$
(\mathbf a\times \mathbf B_0)_\parallel
=
(\mathbf a\times \hat{\mathbf z})_z
=0.
$$

Thus the parallel momentum moment becomes
$$
\partial_t(n_su_{s,\parallel})
=
-\nabla\cdot\boldsymbol{\Pi}_{s,\parallel}
+\alpha_s n_sE_\parallel .
$$

For ions,
$$
\partial_t J_{i,\parallel}
=
-\nabla\cdot\boldsymbol{\Pi}_{i,\parallel}
+n_iE_\parallel .
$$

For electrons,
$$
\partial_t(J_{e,\parallel})
=
-\nabla\cdot\boldsymbol{\Pi}_{e,\parallel}
-\frac{m_i}{m_e}n_eE_\parallel .
$$

Therefore,
$$
\partial_t(J_{i,\parallel}-J_{e,\parallel})
=
\nabla\cdot
(\boldsymbol{\Pi}_{e,\parallel}-\boldsymbol{\Pi}_{i,\parallel})
+
\left(
n_i+\frac{m_i}{m_e}n_e
\right)E_\parallel .
$$

Substitution gives
$$
\boxed{
\left[
\nabla_\perp^2
-
\frac{\beta_i}{2}
\left(
n_i+\frac{m_i}{m_e}n_e
\right)
\right]E_\parallel
=
\frac{\beta_i}{2}
\nabla\cdot
(\boldsymbol{\Pi}_{e,\parallel}-\boldsymbol{\Pi}_{i,\parallel})
+
\partial_\parallel
(\nabla_\perp\cdot\mathbf E_\perp)
}
$$

Equivalently,
$$
\mathcal D E_\parallel=\mathrm{RHS},
$$
with
$$
\mathcal D
=
\nabla_\perp^2
-
\frac{\beta_i}{2}
\left(
n_i+\frac{m_i}{m_e}n_e
\right),
$$
and
$$
\mathrm{RHS}
=
\frac{\beta_i}{2}
\nabla\cdot
(\boldsymbol{\Pi}_{e,\parallel}-\boldsymbol{\Pi}_{i,\parallel})
+
\partial_\parallel
(\nabla_\perp\cdot\mathbf E_\perp).
$$



## Summary
Guide-field model with $\mathbf B_0=\hat{\mathbf z}$, $E_\parallel=E_z$, and $\mu=m_e/m_i$:

$$
\partial_t f_i+\mathbf v\cdot\nabla f_i+
(\mathbf E+\mathbf v\times\mathbf B)\cdot\nabla_{\mathbf v}f_i=0 .
$$

$$
\partial_t f_e+\mathbf u_E\cdot\nabla f_e+v_z\partial_z f_e
-\frac{1}{\mu}E_z\partial_{v_z}f_e=0,
\qquad
\mathbf u_E=\mathbf E\times\hat{\mathbf z}.
$$

Moments:
$$
n_i=\int f_i\,d^3v,\quad \mathbf J_i=\int \mathbf v f_i\,d^3v,
\qquad
n_e=\int f_e\,dv_z,\quad J_{e,\parallel}=\int v_z f_e\,dv_z .
$$

Perpendicular electric field from perpendicular Ampère:
$$
(\partial_t+\mathbf u_E\cdot\nabla)\mathbf E_\perp
=
\frac{1}{\mu n_e}
\left[
\frac{2}{\beta_i}(\nabla\times\mathbf B)_\perp
-\mathbf J_{i,\perp}
+n_e\mathbf E\times\hat{\mathbf z}
\right].
$$

Parallel electric field:
$$
\mathcal D E_z=\mathrm{RHS},
$$
with
$$
\mathcal D=
\nabla_\perp^2
-\frac{\beta_i}{2}
\left(n_i+\frac{1}{\mu}n_e\right),
$$
and
$$
\mathrm{RHS}
=
\frac{\beta_i}{2}
\nabla\cdot
(\boldsymbol{\Pi}_{e,z}-\boldsymbol{\Pi}_{i,z})
+
\partial_z(\nabla_\perp\cdot\mathbf E_\perp).
$$

Here
$$
\boldsymbol{\Pi}_{i,z}=\int v_z\mathbf v f_i\,d^3v,
\qquad
\boldsymbol{\Pi}_{e,z}
=
\mathbf u_E J_{e,\parallel}
+
\hat{\mathbf z}\int v_z^2 f_e\,dv_z .
$$

Magnetic field:
$$
\partial_t\mathbf B=-\nabla\times\mathbf E,
\qquad
\nabla\cdot\mathbf B=0.
$$

Parallel Ampère constraint:
$$
(\nabla\times\mathbf B)_z
=
\frac{\beta_i}{2}
\left(J_{i,z}-J_{e,\parallel}\right).
$$

## Linearization
Perpendicular electric field:
$$
\partial_t\delta\mathbf E_\perp
=
\frac{1}{\mu}
\left[
\frac{2}{\beta_i}
(\nabla\times\delta\mathbf B)_\perp
-
\delta\mathbf J_{i,\perp}
+
\delta\mathbf E_\perp\times\hat{\mathbf z}
\right].
$$

Parallel electric field:
$$
\left[
\nabla_\perp^2
-
\frac{\beta_i}{2}
\left(1+\frac{1}{\mu}\right)
\right]\delta E_z
=
\frac{\beta_i}{2}
\nabla\cdot
\left(
\delta\boldsymbol{\Pi}_{e,z}
-
\delta\boldsymbol{\Pi}_{i,z}
\right)
+
\partial_z
\left(
\nabla_\perp\cdot\delta\mathbf E_\perp
\right).
$$

with
$$
\delta\boldsymbol{\Pi}_{i,z}
=
\int v_z\mathbf v\,\delta f_i\,d^3v,
\qquad
\delta\boldsymbol{\Pi}_{e,z}
=
\hat{\mathbf z}
\int v_z^2\delta f_e\,dv_z .
$$

Magnetic field:
$$
\partial_t\delta\mathbf B
=
-\nabla\times\delta\mathbf E,
\qquad
\nabla\cdot\delta\mathbf B=0 .
$$

Parallel Ampère constraint:
$$
(\nabla\times\delta\mathbf B)_z
=
\frac{\beta_i}{2}
\left(
\delta J_{i,z}
-
\delta J_{e,\parallel}
\right).
$$

## Time discretezation

### Fully explicte field updates

With time step $\Delta t$, treat fields implicitly and particle moments explicitly at time $n$.

Unknowns:
$$
\mathbf E_\perp^{n+1},\qquad E_z^{n+1},\qquad \mathbf B^{n+1}.
$$

Explicit particle moments:
$$
\delta\mathbf J_i^n,\qquad
\delta\boldsymbol{\Pi}_{i,z}^n,\qquad
\delta\boldsymbol{\Pi}_{e,z}^n .
$$

Implicit Euler for $\mathbf E_\perp$:
$$
\frac{\mathbf E_\perp^{n+1}-\mathbf E_\perp^n}{\Delta t}
=
\frac{1}{\mu}
\left[
\frac{2}{\beta_i}
(\nabla\times\mathbf B^{n+1})_\perp
-
\mathbf J_{i,\perp}^n
+
\mathbf E_\perp^{n+1}\times\hat{\mathbf z}
\right].
$$

Implicit Euler for $\mathbf B$:
$$
\frac{\mathbf B^{n+1}-\mathbf B^n}{\Delta t}
=
-\nabla\times
\left(
\mathbf E_\perp^{n+1}
+
E_z^{n+1}\hat{\mathbf z}
\right).
$$

Implicit elliptic solve for $E_z^{n+1}$:
$$
\left[
\nabla_\perp^2
-
\frac{\beta_i}{2}
\left(
1+\frac{1}{\mu}
\right)
\right]
E_z^{n+1}
=
\frac{\beta_i}{2}
\nabla\cdot
\left(
\boldsymbol{\Pi}_{e,z}^n
-
\boldsymbol{\Pi}_{i,z}^n
\right)
+
\partial_z
\left(
\nabla_\perp\cdot\mathbf E_\perp^{n+1}
\right).
$$

Equivalently, eliminating $\mathbf B^{n+1}$:
$$
\mathbf B^{n+1}
=
\mathbf B^n
-
\Delta t\,
\nabla\times
\left(
\mathbf E_\perp^{n+1}
+
E_z^{n+1}\hat{\mathbf z}
\right),
$$

so
$$
\frac{\mathbf E_\perp^{n+1}-\mathbf E_\perp^n}{\Delta t}
=
\frac{1}{\mu}
\left[
\frac{2}{\beta_i}
\left(
\nabla\times\mathbf B^n
-
\Delta t\,\nabla\times\nabla\times\mathbf E^{n+1}
\right)_\perp
-
\mathbf J_{i,\perp}^n
+
\mathbf E_\perp^{n+1}\times\hat{\mathbf z}
\right],
$$


### Semi-implicit field update with explicit Faraday law

Particle moments are explicit:
$$
\mathbf J_{i,\perp}^n,\qquad
\boldsymbol{\Pi}_{i,z}^n,\qquad
\boldsymbol{\Pi}_{e,z}^n .
$$

Faraday's law is treated explicitly:
$$
\mathbf B^{n+1}
=
\mathbf B^n
-
\Delta t\,\nabla\times
\left(
\mathbf E_\perp^n+E_z^n\hat{\mathbf z}
\right).
$$

The perpendicular electric field is updated implicitly:
$$
\frac{\mathbf E_\perp^{n+1}-\mathbf E_\perp^n}{\Delta t}
=
\frac{1}{\mu}
\left[
\frac{2}{\beta_i}
(\nabla\times\mathbf B^{n+1})_\perp
-
\mathbf J_{i,\perp}^n
+
\mathbf E_\perp^{n+1}\times\hat{\mathbf z}
\right].
$$

Equivalently,
$$
\left[
\mathbf I
-
\frac{\Delta t}{\mu}
\mathbf R
\right]\mathbf E_\perp^{n+1}
=
\mathbf E_\perp^n
+
\frac{\Delta t}{\mu}
\left[
\frac{2}{\beta_i}
(\nabla\times\mathbf B^{n+1})_\perp
-
\mathbf J_{i,\perp}^n
\right],
$$
where $\mathbf R\mathbf a=\mathbf a\times\hat{\mathbf z}$.

The parallel electric field is then solved from
$$
\left[
\nabla_\perp^2
-
\frac{\beta_i}{2}
\left(
1+\frac{1}{\mu}
\right)
\right]
E_z^{n+1}
=
\frac{\beta_i}{2}
\nabla\cdot
\left(
\boldsymbol{\Pi}_{e,z}^n
-
\boldsymbol{\Pi}_{i,z}^n
\right)
+
\partial_z
\left(
\nabla_\perp\cdot\mathbf E_\perp^{n+1}
\right).
$$




## Step-by-step implementation plan

1. Add staggered magnetic field storage.

   Store $E^n$ and $B^{n-1/2}$.

   Recommended fields in `FieldSolution`:

   ```julia
   sol.E        # E^n, overwritten by E^{n+1}
   sol.Eold     # buffer for E^n
   sol.Ecenter  # E^{n+1/2}
   sol.Bhalf    # B^{n-1/2}, overwritten by B^{n+1/2}
   ```

2. Initialize `Bhalf`.

   If the code currently has $B^0$, initialize

   $$
   B^{-1/2}
   =
   B^0
   +
   \frac{\Delta t}{2}
   \nabla\times E^0 .
   $$

   In code:

   ```julia
   curlE = curl(sol.E, grid)
   for d in 1:ncomponents(sol.Bhalf)
       sol.Bhalf[d].data .= sol.B[d].data .+ 0.5 * dt .* curlE[d].data
   end
   ```

3. Modify the semi-implicit field solver.

   The solver should use:

   ```julia
   sol.E      # input E^n, output E^{n+1}
   sol.Bhalf  # input B^{n-1/2}, output B^{n+1/2}
   ```

   Replace all use of `sol.B` inside the semi-implicit EM solver by `sol.Bhalf`.

4. Implement explicit staggered Faraday update.

   Update

   $$
   B^{n+1/2}
   =
   B^{n-1/2}
   -
   \Delta t
   \nabla\times E^n .
   $$

   In code:

   ```julia
   curlE = curl(E, grid)
   for d in 1:ncomponents(Bhalf)
       Bhalf[d].data .-= dt .* curlE[d].data
   end
   ```

5. Implement the perpendicular implicit electric-field update.

   After the Faraday step, compute

   $$
   \nabla\times B^{n+1/2}.
   $$

   Then solve

   $$
   \left[
   I
   -
   \frac{\Delta t}{\mu}
   R
   \right]
   E_\perp^{n+1}
   =
   E_\perp^n
   +
   \frac{\Delta t}{\mu}
   \left[
   \frac{2}{\beta_i}
   \left(
   \nabla\times B^{n+1/2}
   \right)_\perp
   -
   J_{i,\perp}^n
   \right].
   $$

   With $R(a_1,a_2)=(a_2,-a_1)$, the local solve is

   ```julia
   α = dt / solver.mu
   c = 2 / solver.beta_i

   rhs1 = E[d1].data .+ α .* (c .* curlB[d1].data .- J_i_perp[d1].data)
   rhs2 = E[d2].data .+ α .* (c .* curlB[d2].data .- J_i_perp[d2].data)

   denom = 1 + α^2

   E[d1].data .= (rhs1 .+ α .* rhs2) ./ denom
   E[d2].data .= (rhs2 .- α .* rhs1) ./ denom
   ```

6. Implement the parallel Helmholtz solve.

   Use the updated $E_\perp^{n+1}$ and solve

   $$
   \left[
   \nabla_\perp^2
   -
   \frac{\beta_i}{2}
   \left(
   1+\frac{1}{\mu}
   \right)
   \right]
   E_z^{n+1}
   =
   \frac{\beta_i}{2}
   \nabla\cdot
   \left(
   \Pi_{e,z}^n
   -
   \Pi_{i,z}^n
   \right)
   +
   \partial_z
   \left(
   \nabla_\perp\cdot E_\perp^{n+1}
   \right).
   $$

   This part can mostly stay as in your current implementation.

7. Add centered electric field construction.

   Before the field update, store $E^n$:

   ```julia
   copy!(sol.Eold, sol.E)
   ```

   After the field update, construct

   $$
   E^{n+1/2}
   =
   \frac12
   \left(
   E^n+E^{n+1}
   \right).
   $$

   In code:

   ```julia
   for d in 1:ncomponents(sol.E)
       sol.Ecenter[d].data .= 0.5 .* (sol.Eold[d].data .+ sol.E[d].data)
   end
   ```

8. Change the full timestep order.

   Each timestep should be:

   ```text
   given f_i^n, E^n, B^{n-1/2}

   1. compute moments from f_i^n
   2. save E^n into Eold
   3. update B^{n-1/2} -> B^{n+1/2}
   4. update E^n -> E^{n+1}
   5. build E^{n+1/2}
   6. advance particles f_i^n -> f_i^{n+1}
      using frozen E^{n+1/2}, B^{n+1/2}
   7. exit with E^{n+1}, B^{n+1/2}
   ```

9. Implement the Strang step.

   ```julia
   function stepStrangSemiImplicitEM!(f_i, sol, grid, solver, simTime)

       phase_start = simTime.phase
       Ω  = simTime.gyro_frequency
       dt = simTime.dt

       # ------------------------------------------------------------
       # Input state:
       #   f_i       = f_i^n
       #   sol.E     = E^n
       #   sol.Bhalf = B^{n-1/2}
       # ------------------------------------------------------------

       # moments from f_i^n
       n_i       = bslLD.compute_density(f_i, grid)
       J_i_perp  = compute_J_perp(f_i, grid, simTime)
       Pi_diff_z = compute_Pi_diff_z(f_i, grid, simTime)

       moments = bslLD.Moments(n_i, J_i_perp, Pi_diff_z)

       # save E^n
       copy!(sol.Eold, sol.E)

       # field update:
       #   B^{n-1/2} -> B^{n+1/2}
       #   E^n       -> E^{n+1}
       bslLD.solve_fields!(sol, moments, grid, solver, dt)

       # centered electric field
       for d in 1:ncomponents(sol.E)
           sol.Ecenter[d].data .= 0.5 .* (sol.Eold[d].data .+ sol.E[d].data)
       end

       # V half-step with E^{n+1/2}, B^{n+1/2}
       simTime.phase = phase_start
       simTime.fraction_dt = 0.5
       bslLD.advectV!(f_i, grid, simTime, sol.Ecenter, sol.Bhalf)

       # X full-step
       simTime.phase = phase_start + 0.5 * Ω * dt
       simTime.fraction_dt = 1.0
       bslLD.advectX!(f_i, grid, simTime)

       # V half-step with same frozen fields
       simTime.phase = phase_start + Ω * dt
       simTime.fraction_dt = 0.5
       bslLD.advectV!(f_i, grid, simTime, sol.Ecenter, sol.Bhalf)

       # restore bookkeeping
       simTime.phase = phase_start
       simTime.fraction_dt = 1.0

       # ------------------------------------------------------------
       # Output state:
       #   f_i       = f_i^{n+1}
       #   sol.E     = E^{n+1}
       #   sol.Bhalf = B^{n+1/2}
       # ------------------------------------------------------------

       return nothing
   end
   ```

10. Modify `solve_fields!`.

   ```julia
   function solve_fields!(
       sol::FieldSolution,
       moments::Moments,
       grid::Grid,
       solver::SemiImplicitEMSolver,
       dt::Real,
   )
       moments.J !== nothing ||
           throw(ArgumentError("moments.J is required"))

       moments.Pi_diff_z !== nothing ||
           throw(ArgumentError("moments.Pi_diff_z is required"))

       grid.Bdir == 3 ||
           throw(ArgumentError("SemiImplicitEMSolver requires grid.Bdir == 3"))

       _step_semi_implicit_em!(
           sol.E,
           sol.Bhalf,
           moments.J,
           moments.Pi_diff_z,
           grid,
           solver,
           dt,
       )

       return sol
   end
   ```

11. Rename the internal field-step arguments.

   Prefer

   ```julia
   function _step_semi_implicit_em!(
       E::VectorField,
       Bhalf::VectorField,
       J_i_perp::VectorField,
       Pi_diff_z::VectorField,
       grid::Grid,
       solver,
       dt::Real,
   )
   ```

   instead of calling the magnetic field simply `B`.

12. Check the time-level consistency.

   At timestep entry:

   ```text
   sol.E     = E^n
   sol.Bhalf = B^{n-1/2}
   f_i       = f_i^n
   ```

   After field solve:

   ```text
   sol.E     = E^{n+1}
   sol.Bhalf = B^{n+1/2}
   ```

   During particle push:

   ```text
   use sol.Ecenter = E^{n+1/2}
   use sol.Bhalf   = B^{n+1/2}
   ```

   At timestep exit:

   ```text
   sol.E     = E^{n+1}
   sol.Bhalf = B^{n+1/2}
   f_i       = f_i^{n+1}
   ```

13. Add basic tests.

   Test at least:

   - zero-field equilibrium
   - uniform-field case
   - single Fourier mode
   - timestep refinement

   For timestep refinement, check whether the error behaves approximately like

   $$
   \mathcal O(\Delta t^2)
   $$

   for the full Strang-coupled scheme.

14. Important note.

   Do not update moments again inside the same timestep before the particle push.
   The field update uses explicit moments from $f_i^n$, and the particle step then advances with frozen centered fields.




   # Model without electron polarization drift

Define $\mu=m_e/m_i$, hence $m_i/m_e=1/\mu$. The only change is
$$\mathbf J_{e,p}=0.$$

## Derivation

The electron current is
$$\mathbf J_e=-n_e\mathbf E\times\hat{\mathbf z}-n_eu_{e,z}\hat{\mathbf z}.$$

Thus Ampère's law is
$$\nabla\times\mathbf B=\frac{\beta_i}{2}\left[\mathbf J_i-n_e\mathbf E\times\hat{\mathbf z}-n_eu_{e,z}\hat{\mathbf z}\right].$$

The perpendicular part gives
$$\left(\nabla\times\mathbf B\right)_\perp=\frac{\beta_i}{2}\left[\mathbf J_{i,\perp}-n_e\mathbf E_\perp\times\hat{\mathbf z}\right].$$

Solving for $\mathbf E_\perp$,
$$\boxed{\mathbf E_\perp=\frac{1}{n_e}\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B\right)_\perp-\mathbf J_{i,\perp}\right]\times\hat{\mathbf z}}.$$

The parallel part is unchanged:
$$\left(\nabla\times\mathbf B\right)_z=\frac{\beta_i}{2}\left(J_{i,z}-n_eu_{e,z}\right).$$

Using Faraday's law,
$$\partial_t\mathbf B=-\nabla\times\mathbf E,$$
one obtains
$$\nabla_\perp^2E_z=\frac{\beta_i}{2}\partial_t\left(J_{i,z}-J_{e,z}\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp\right).$$

The parallel momentum moments give
$$\partial_t\left(J_{i,z}-J_{e,z}\right)=\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}-\boldsymbol{\Pi}_{i,z}\right)+\left(n_i+\frac{1}{\mu}n_e\right)E_z.$$

Therefore
$$\boxed{\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(n_i+\frac{1}{\mu}n_e\right)\right]E_z=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}-\boldsymbol{\Pi}_{i,z}\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp\right)}.$$

For $n_i=n_e=1$,
$$\boxed{\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]E_z=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}-\boldsymbol{\Pi}_{i,z}\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp\right)}.$$

## Summary

With electron polarization drift neglected:

- $\mathbf E_\perp$ is no longer evolved by an implicit polarization equation.
- $\mathbf E_\perp$ is obtained algebraically from perpendicular Ampère's law.
- $E_z$ is still obtained from the same Helmholtz equation.
- Faraday's law still updates $\mathbf B$.

The field closure is
$$\mathbf E_\perp=\frac{1}{n_e}\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B\right)_\perp-\mathbf J_{i,\perp}\right]\times\hat{\mathbf z},$$
$$\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(n_i+\frac{1}{\mu}n_e\right)\right]E_z=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}-\boldsymbol{\Pi}_{i,z}\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp\right).$$

## Linearization

Linearize around
$$n_i=n_e=1,\qquad \mathbf E=0,\qquad \mathbf J_i=0,\qquad \mathbf B=\hat{\mathbf z}+\delta\mathbf B.$$

Then
$$\boxed{\delta\mathbf E_\perp=\left[\frac{2}{\beta_i}\left(\nabla\times\delta\mathbf B\right)_\perp-\delta\mathbf J_{i,\perp}\right]\times\hat{\mathbf z}}.$$

The linearized parallel equation is
$$\boxed{\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]\delta E_z=\frac{\beta_i}{2}\nabla\cdot\left(\delta\boldsymbol{\Pi}_{e,z}-\delta\boldsymbol{\Pi}_{i,z}\right)+\partial_z\left(\nabla_\perp\cdot\delta\mathbf E_\perp\right)}.$$

For a Fourier mode $\sim e^{i\mathbf k\cdot\mathbf x}$,
$$\delta\mathbf E_{\perp,\mathbf k}=\left[\frac{2i}{\beta_i}\left(\mathbf k\times\delta\mathbf B_{\mathbf k}\right)_\perp-\delta\mathbf J_{i,\perp,\mathbf k}\right]\times\hat{\mathbf z},$$
$$\left[-k_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]\delta E_{z,\mathbf k}=\frac{i\beta_i}{2}\mathbf k\cdot\left(\delta\boldsymbol{\Pi}_{e,z,\mathbf k}-\delta\boldsymbol{\Pi}_{i,z,\mathbf k}\right)-k_z\mathbf k_\perp\cdot\delta\mathbf E_{\perp,\mathbf k}.$$

## Time discretization

### Fully implicit field update

Unknowns:
$$\mathbf E_\perp^{n+1},\qquad E_z^{n+1},\qquad \mathbf B^{n+1}.$$

Explicit moments:
$$\mathbf J_{i,\perp}^n,\qquad \boldsymbol{\Pi}_{i,z}^n,\qquad \boldsymbol{\Pi}_{e,z}^n.$$

Perpendicular Ampère is algebraic:
$$\boxed{\mathbf E_\perp^{n+1}=\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B^{n+1}\right)_\perp-\mathbf J_{i,\perp}^n\right]\times\hat{\mathbf z}}.$$

Faraday:
$$\frac{\mathbf B^{n+1}-\mathbf B^n}{\Delta t}=-\nabla\times\left(\mathbf E_\perp^{n+1}+E_z^{n+1}\hat{\mathbf z}\right).$$

Parallel Helmholtz solve:
$$\boxed{\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]E_z^{n+1}=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp^{n+1}\right)}.$$

Eliminating $\mathbf B^{n+1}$,
$$\mathbf B^{n+1}=\mathbf B^n-\Delta t\nabla\times\left(\mathbf E_\perp^{n+1}+E_z^{n+1}\hat{\mathbf z}\right),$$
so
$$\boxed{\mathbf E_\perp^{n+1}=\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B^n-\Delta t\nabla\times\nabla\times\mathbf E^{n+1}\right)_\perp-\mathbf J_{i,\perp}^n\right]\times\hat{\mathbf z}}.$$

### Semi-implicit field update with explicit Faraday law

Particle moments are explicit:
$$\mathbf J_{i,\perp}^n,\qquad \boldsymbol{\Pi}_{i,z}^n,\qquad \boldsymbol{\Pi}_{e,z}^n.$$

Explicit Faraday:
$$\mathbf B^{n+1}=\mathbf B^n-\Delta t\nabla\times\left(\mathbf E_\perp^n+E_z^n\hat{\mathbf z}\right).$$

Then update $\mathbf E_\perp$ algebraically:
$$\boxed{\mathbf E_\perp^{n+1}=\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B^{n+1}\right)_\perp-\mathbf J_{i,\perp}^n\right]\times\hat{\mathbf z}}.$$

Equivalently, with $R\mathbf a=\mathbf a\times\hat{\mathbf z}$,
$$\mathbf E_\perp^{n+1}=R\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B^{n+1}\right)_\perp-\mathbf J_{i,\perp}^n\right].$$

Then solve
$$\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]E_z^{n+1}=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp^{n+1}\right).$$

### Staggered version

Store $E^n$ and $B^{n-1/2}$. Faraday becomes
$$B^{n+1/2}=B^{n-1/2}-\Delta t\nabla\times E^n.$$

Then
$$\boxed{\mathbf E_\perp^{n+1}=\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B^{n+1/2}\right)_\perp-\mathbf J_{i,\perp}^n\right]\times\hat{\mathbf z}}.$$

Finally solve
$$\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]E_z^{n+1}=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp^{n+1}\right).$$

## Implementation plan

1. Store staggered fields:
   - `sol.E = E^n`
   - `sol.Eold = E^n` buffer
   - `sol.Ecenter = E^{n+1/2}`
   - `sol.Bhalf = B^{n-1/2}`

2. Initialize `Bhalf` from $B^0$:
$$B^{-1/2}=B^0+\frac{\Delta t}{2}\nabla\times E^0.$$

3. At each timestep, compute explicit moments from $f_i^n$:
$$\mathbf J_{i,\perp}^n,\qquad \boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n.$$

4. Save old electric field:
    copy!(sol.Eold, sol.E)

5. Update magnetic field explicitly:
$$B^{n+1/2}=B^{n-1/2}-\Delta t\nabla\times E^n.$$

6. Compute $\nabla\times B^{n+1/2}$.

7. Replace the old implicit perpendicular solve by the algebraic update:
$$\mathbf E_\perp^{n+1}=\left[\frac{2}{\beta_i}\left(\nabla\times\mathbf B^{n+1/2}\right)_\perp-\mathbf J_{i,\perp}^n\right]\times\hat{\mathbf z}.$$

In code, if $R(a_1,a_2)=(a_2,-a_1)$:

    c = 2 / solver.beta_i
    rhs1 = c .* curlB[d1].data .- J_i_perp[d1].data
    rhs2 = c .* curlB[d2].data .- J_i_perp[d2].data
    E[d1].data .= rhs2
    E[d2].data .= -rhs1

8. Solve the parallel Helmholtz equation for $E_z^{n+1}$:
$$\left[\nabla_\perp^2-\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right)\right]E_z^{n+1}=\frac{\beta_i}{2}\nabla\cdot\left(\boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n\right)+\partial_z\left(\nabla_\perp\cdot\mathbf E_\perp^{n+1}\right).$$

9. Build centered electric field:
$$E^{n+1/2}=\frac{1}{2}\left(E^n+E^{n+1}\right).$$

10. Push particles with frozen fields:
$$E=E^{n+1/2},\qquad B=B^{n+1/2}.$$

11. Timestep order:
    given f_i^n, E^n, B^{n-1/2}
    1. compute moments from f_i^n
    2. save E^n
    3. update B^{n-1/2} -> B^{n+1/2}
    4. compute E_perp^{n+1} algebraically
    5. solve E_z^{n+1}
    6. build E^{n+1/2}
    7. push particles f_i^n -> f_i^{n+1}
    8. exit with f_i^{n+1}, E^{n+1}, B^{n+1/2}

12. Important difference from the polarization model:
$$\left[I-\frac{\Delta t}{\mu}R\right]\mathbf E_\perp^{n+1}=\cdots$$
is removed completely. The perpendicular electric field is now obtained directly from Ampère's law.




## Fully implicit Fourier implementation strategy

Assume a periodic domain and constant densities
$$n_i=n_e=1.$$

Define
$$c=\frac{2}{\beta_i},\qquad \lambda=\frac{\beta_i}{2}\left(1+\frac{1}{\mu}\right).$$

At timestep $n$, known quantities are
$$\mathbf E^n,\qquad \mathbf B^n,\qquad \mathbf J_{i,\perp}^n,\qquad \boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n.$$

Unknowns are
$$E_x^{n+1},\qquad E_y^{n+1},\qquad E_z^{n+1}.$$

After solving for $\mathbf E^{n+1}$, update
$$\mathbf B^{n+1}=\mathbf B^n-\Delta t\nabla\times\mathbf E^{n+1}.$$

### 1. Transform known fields to Fourier space

For each Fourier mode $\mathbf k=(k_x,k_y,k_z)$, compute
$$\widehat{\mathbf B}^n_{\mathbf k},\qquad \widehat{\mathbf J}_{i,\perp,\mathbf k}^n,\qquad \widehat{\mathbf Q}_{z,\mathbf k}^n,$$
where
$$\mathbf Q_z^n=\boldsymbol{\Pi}_{e,z}^n-\boldsymbol{\Pi}_{i,z}^n.$$

### 2. Build the right-hand side

The perpendicular RHS is
$$\mathbf b_{\perp,\mathbf k}=R\left[c\left(i\mathbf k\times\widehat{\mathbf B}^n_{\mathbf k}\right)_\perp-\widehat{\mathbf J}_{i,\perp,\mathbf k}^n\right],$$
where
$$R(a_x,a_y)=(a_y,-a_x).$$

The parallel RHS is
$$b_{z,\mathbf k}=\frac{i\beta_i}{2}\mathbf k\cdot\widehat{\mathbf Q}_{z,\mathbf k}^n.$$

Thus
$$\mathbf b_{\mathbf k}=\begin{pmatrix}b_{x,\mathbf k}\\b_{y,\mathbf k}\\b_{z,\mathbf k}\end{pmatrix}.$$

### 3. Build the $3\times3$ Fourier operator

Use
$$\nabla\times\nabla\times\mathbf E\to k^2\mathbf E-\mathbf k(\mathbf k\cdot\mathbf E),$$
with
$$k^2=k_x^2+k_y^2+k_z^2,\qquad k_\perp^2=k_x^2+k_y^2.$$

Define
$$\mathbf C_\perp=\left(k^2\mathbf E-\mathbf k(\mathbf k\cdot\mathbf E)\right)_\perp.$$

The perpendicular equations are
$$\mathbf E_\perp+c\Delta t\,R\mathbf C_\perp=\mathbf b_\perp.$$

The parallel equation is
$$(-k_\perp^2-\lambda)E_z+k_z(k_xE_x+k_yE_y)=b_z.$$

Therefore solve
$$A(\mathbf k)\widehat{\mathbf E}_{\mathbf k}^{n+1}=\mathbf b_{\mathbf k}.$$

The matrix entries are

$$A_{11}=1+c\Delta t\,(-k_xk_y),$$
$$A_{12}=c\Delta t\,(k^2-k_y^2),$$
$$A_{13}=c\Delta t\,(-k_yk_z),$$

$$A_{21}=-c\Delta t\,(k^2-k_x^2),$$
$$A_{22}=1+c\Delta t\,(k_xk_y),$$
$$A_{23}=c\Delta t\,(k_xk_z),$$

$$A_{31}=k_zk_x,$$
$$A_{32}=k_zk_y,$$
$$A_{33}=-k_\perp^2-\lambda.$$

### 4. Solve each Fourier mode independently

For every $\mathbf k$, solve
$$\widehat{\mathbf E}_{\mathbf k}^{n+1}=A(\mathbf k)^{-1}\mathbf b_{\mathbf k}.$$

In code this is just a local $3\times3$ complex linear solve per mode.

### 5. Update magnetic field in Fourier space

After $\widehat{\mathbf E}^{n+1}_{\mathbf k}$ is known, update
$$\widehat{\mathbf B}^{n+1}_{\mathbf k}=\widehat{\mathbf B}^n_{\mathbf k}-\Delta t\,i\mathbf k\times\widehat{\mathbf E}^{n+1}_{\mathbf k}.$$

### 6. Transform back to real space

Apply inverse FFT:
$$\mathbf E^{n+1}=\mathcal F^{-1}\left[\widehat{\mathbf E}^{n+1}\right],\qquad \mathbf B^{n+1}=\mathcal F^{-1}\left[\widehat{\mathbf B}^{n+1}\right].$$

### 7. Timestep order

```text
given f_i^n, E^n, B^n

1. compute explicit moments from f_i^n:
   J_i_perp^n, Pi_diff_z^n

2. FFT B^n, J_i_perp^n, Pi_diff_z^n

3. for each Fourier mode k:
   a. build RHS b_k
   b. build 3x3 matrix A(k)
   c. solve A(k) E_k^{n+1} = b_k
   d. update B_k^{n+1} = B_k^n - dt i k x E_k^{n+1}

4. inverse FFT E^{n+1}, B^{n+1}

5. construct centered field:
   E^{n+1/2} = 0.5(E^n + E^{n+1})

6. push particles with frozen fields:
   E^{n+1/2}, B^{n+1}

7. exit with f_i^{n+1}, E^{n+1}, B^{n+1}
```

### 8. Important restrictions

This Fourier diagonalization works only if:

- the domain is periodic,
- the coefficients are constant,
- typically $n_i=n_e=1$,
- the guide field direction is fixed,
- moments are treated explicitly.

If $n_i$ or $n_e$ vary spatially, Fourier modes couple and this becomes a global variable-coefficient solve.