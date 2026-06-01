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