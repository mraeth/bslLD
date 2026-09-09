# Local toroidal full-orbit Vlasov model with a logical rotating velocity grid

## 1. Normalization

For singly charged ions, define

$
\Omega_{i0}=\frac{eB_{\mathrm{ref}}}{m_i},
\qquad
v_{ti0}=\sqrt{\frac{T_{i0}}{m_i}},
\qquad
\rho_{i0}=\frac{v_{ti0}}{\Omega_{i0}}.
$

Use dimensionless variables

$
t=\Omega_{i0}t_{\mathrm{phys}},
\qquad
\mathbf x=\frac{\mathbf x_{\mathrm{phys}}}{\rho_{i0}},
\qquad
\mathbf v=\frac{\mathbf v_{\mathrm{phys}}}{v_{ti0}},
$

$
\mathbf B=\frac{\mathbf B_{\mathrm{phys}}}{B_{\mathrm{ref}}},
\qquad
\phi=\frac{e\Phi}{T_{i0}},
\qquad
\mathbf E=-\nabla\phi.
$

All radii and scale lengths below, including $R_0$, $r_0$, $a$, $L_x$, $L_y$, and $L_z$, are therefore measured in units of $\rho_{i0}$.

---

## 2. Global geometry and local coordinates

Consider circular tokamak flux surfaces with major radius $R_0$, minor radius $r$, poloidal angle $\theta$, and toroidal angle $\varphi$:

$
R(r,\theta)=R_0+r\cos\theta.
$

Choose a reference point

$
(r,\theta,\varphi)=(r_0,\theta_0,\varphi_0)
$

and define the local coordinates

$
x=r-r_0,
\qquad
y=r_0(\theta-\theta_0),
\qquad
z=R_c(\varphi-\varphi_0),
$

where

$
R_c=R_0+r_0\cos\theta_0.
$

The directions are:

- $x$: minor-radial;
- $y$: local poloidal;
- $z$: local toroidal.

For a centered box,

$
-\frac{L_x}{2}\leq x<\frac{L_x}{2},
\qquad
-\frac{L_y}{2}\leq y<\frac{L_y}{2},
\qquad
-\frac{L_z}{2}\leq z<\frac{L_z}{2}.
$

Its location in the global torus is

$
r=r_0+x,
\qquad
\theta=\theta_0+\frac{y}{r_0},
\qquad
\varphi=\varphi_0+\frac{z}{R_c}.
$

The local approximation requires

$
L_x\ll r_0,
\qquad
L_y\ll 2\pi r_0,
\qquad
L_z\ll 2\pi R_c.
$

For a local bad-curvature model, choose the outboard midplane:

$
\theta_0=0,
\qquad
R_c=R_0+r_0.
$

The poloidal dependence of the geometry is then frozen at this point. The periodic $y$-domain represents a repeated local patch, not a complete poloidal circuit.

---

## 3. Magnetic field, safety factor, and shear

Write the local magnetic field as

$
\mathbf B(x)=B_y(x)\hat{\mathbf y}+B_z(x)\hat{\mathbf z}.
$

Using the toroidal field at the box center as the reference field,

$
B_{\mathrm{ref}}=B_z(0),
$

a large-aspect-ratio approximation is

$
B_z(x)=\frac{R_c}{R_c+x}
      \simeq 1-\frac{x}{R_c}.
$

The poloidal component is determined by the safety factor:

$
\frac{B_y(x)}{B_z(x)}
\simeq
\frac{r_0+x}{q(x)R_0},
$

with

$
q(x)=q_0\left(1+\hat s\frac{x}{r_0}\right),
\qquad
\hat s=
\frac{r_0}{q_0}
\left.\frac{dq}{dr}\right|_{r_0}.
$

Thus:

- $q_0$ determines the local magnetic-field pitch;
- $\hat s$ determines its radial variation;
- $R_c^{-1}$ determines the dominant toroidal curvature;
- $B_z'(x)$ provides the local grad-$B$ effect.

The global parallel connection length is approximately

$
L_\parallel\simeq 2\pi q_0R_0,
$

but a short periodic poloidal patch does not contain this complete connection length.

---

## 4. Physical full-orbit equations

Use physical orthonormal velocity components

$
\mathbf v=(v_x,v_y,v_z).
$

The local full-orbit characteristics are

$
\dot{\mathbf x}=\mathbf v,
$

$
\dot{\mathbf v}
=
\mathbf E
+\mathbf v\times\mathbf B
+\mathbf a_g.
$

At the outboard midplane, the leading local geometric acceleration is

$
a_{g,x}
=
\frac{v_y^2}{r_0}
+\frac{v_z^2}{R_c},
$

$
a_{g,y}
=
-\frac{v_xv_y}{r_0},
\qquad
a_{g,z}
=
-\frac{v_xv_z}{R_c}.
$

The $R_c^{-1}$ terms represent dominant toroidal curvature. The $r_0^{-1}$ terms are associated with the poloidal coordinate curvature and may be omitted in a reduced large-aspect-ratio model.

The physical Vlasov equation is

$
\partial_t f
+\mathbf v\cdot\nabla_{\mathbf x}f
+
\left(
\mathbf E+\mathbf v\times\mathbf B+\mathbf a_g
\right)\cdot\nabla_{\mathbf v}f
=S[f].
$

---

## 5. Logical rotating velocity grid

The implemented slab algorithm removes the dominant cyclotron rotation by introducing a logical velocity coordinate $\mathbf u$. The same method can be retained in the toroidal model.

Choose the constant reference field

$
\mathbf B_0=\hat{\mathbf z},
\qquad
\delta\mathbf B(\mathbf x)=\mathbf B(\mathbf x)-\mathbf B_0.
$

Define

$
\mathbf v=\mathsf Q(t)\mathbf u,
$

where

$
\dot{\mathsf Q}=\mathsf L_0\mathsf Q,
\qquad
\mathsf L_0\mathbf v=\mathbf v\times\mathbf B_0.
$

For $\mathbf B_0=\hat{\mathbf z}$,

$
\mathsf Q(t)=
\begin{pmatrix}
\cos t & \sin t & 0\\
-\sin t & \cos t & 0\\
0&0&1
\end{pmatrix}.
$

Define the distribution on the fixed logical grid by

$
g(\mathbf x,\mathbf u,t)
=
f\!\left(\mathbf x,\mathsf Q(t)\mathbf u,t\right).
$

Because $\det\mathsf Q=1$,

$
d^3v=d^3u,
\qquad
n_i=\int g\,d^3u.
$

The transformed characteristics are

$
\dot{\mathbf x}=\mathsf Q(t)\mathbf u,
$

$
\dot{\mathbf u}
=
\mathsf Q^T(t)
\left[
\mathbf E
+\mathbf a_g\!\left(\mathsf Q\mathbf u\right)
+\left(\mathsf Q\mathbf u\right)\times\delta\mathbf B
\right].
$

Therefore,

$
\partial_t g
+
\left(\mathsf Q\mathbf u\right)\cdot\nabla_{\mathbf x}g
+
\mathbf c_u\cdot\nabla_{\mathbf u}g
=S[g],
$

where

$
\mathbf c_u
=
\mathsf Q^T
\left[
\mathbf E
+\mathbf a_g(\mathsf Q\mathbf u)
+\left(\mathsf Q\mathbf u\right)\times\delta\mathbf B
\right].
$

There is no separate magnetic rotation or velocity-grid remapping. The logical grid remains fixed, while $\mathsf Q(t)$ appears in the advection coefficients.

---

## 6. Quasineutral electric field

The ion density is

$
n_i(\mathbf x,t)=\int g(\mathbf x,\mathbf u,t)\,d^3u.
$

For adiabatic electrons,

$
n_e=n_{e0}(x)\exp\left(\frac{\phi}{\tau}\right),
\qquad
\tau=\frac{T_e}{T_{i0}}.
$

Quasineutrality gives

$
\phi
=
\tau\ln\left(\frac{n_i}{n_{e0}(x)}\right),
\qquad
\mathbf E=-\nabla\phi.
$

A modified adiabatic response may be used for zonal modes, for example by removing the perpendicular average of $\phi$ from the electron response.

---

## 7. Equilibrium gradients and source

A local Maxwellian equilibrium is

$
F_0(x,\mathbf v)
=
\frac{n_0(x)}
{[2\pi T_i(x)]^{3/2}}
\exp\left[-\frac{|\mathbf v|^2}{2T_i(x)}\right].
$

On the logical grid,

$
|\mathbf v|^2=|\mathsf Q\mathbf u|^2=|\mathbf u|^2,
$

so

$
F_0(x,\mathbf u)
=
\frac{n_0(x)}
{[2\pi T_i(x)]^{3/2}}
\exp\left[-\frac{|\mathbf u|^2}{2T_i(x)}\right].
$

The gradient parameters are

$
\frac{a}{L_{T_i}}
=
-a\frac{d\ln T_i}{dx},
\qquad
\frac{a}{L_n}
=
-a\frac{d\ln n_0}{dx},
\qquad
\eta_i=\frac{L_n}{L_{T_i}}.
$

A buffer or Krook source may maintain the profiles:

$
S[g]=-\nu_b(x)\left[g-F_0\right].
$

Without such a source, turbulent transport relaxes the initial gradients.

---

## 8. One-dimensional semi-Lagrangian interpolation

For

$
\partial_t g+c\,\partial_\xi g=0,
$

with $c$ independent of the interpolated coordinate $\xi$, define

$
\mathcal A_\xi(h;c)g
=
\mathcal I_\xi[g](\xi-hc).
$

Here $\mathcal I_\xi$ is a one-dimensional interpolation.

### Spatial coefficients

The physical velocity is evaluated from

$
\mathbf v(t,\mathbf u)=\mathsf Q(t)\mathbf u.
$

Thus

$
c_x=(\mathsf Q\mathbf u)_x,
\qquad
c_y=(\mathsf Q\mathbf u)_y,
\qquad
c_z=(\mathsf Q\mathbf u)_z.
$

The spatial operators are

$
\mathcal X(h)=\mathcal A_x(h;c_x),
\qquad
\mathcal Y(h)=\mathcal A_y(h;c_y),
\qquad
\mathcal Z(h)=\mathcal A_z(h;c_z).
$

Because $\mathsf Q(t)$ changes during a finite substep, the exact spatial displacement may also be used:

$
\Delta\mathbf x
=
\int_{t_a}^{t_b}\mathsf Q(s)\mathbf u\,ds.
$

Using this integral is preferable to evaluating $\mathsf Q$ only at one endpoint.

A symmetric spatial sweep is

$
\mathcal D(h)
=
\mathcal X(h/2)
\mathcal Y(h/2)
\mathcal Z(h)
\mathcal Y(h/2)
\mathcal X(h/2).
$

### Logical-velocity coefficients

The logical-velocity acceleration is

$
\mathbf c_u
=
\mathbf c_E+\mathbf c_B+\mathbf c_g,
$

with

$
\mathbf c_E=\mathsf Q^T\mathbf E,
$

$
\mathbf c_B=
\mathsf Q^T
\left[
(\mathsf Q\mathbf u)\times\delta\mathbf B
\right],
$

$
\mathbf c_g=
\mathsf Q^T
\mathbf a_g(\mathsf Q\mathbf u).
$

The electric coefficients are independent of $\mathbf u$, so they are ordinary translations:

$
\mathcal E_i(h)
=
\mathcal A_{u_i}(h;c_{E,i}).
$

The residual magnetic terms are linear velocity-space shears and can be split into consecutive one-dimensional interpolations.

The geometric terms are quadratic in the physical velocities. After transformation by $\mathsf Q$, a component $c_{g,i}$ can depend on its own logical coordinate $u_i$. Such a subflow is not generally a constant-coefficient translation. It must be handled by one of the following:

1. solve its one-dimensional characteristic exactly and interpolate at the resulting departure point;
2. decompose the geometric map into exact shear or scaling subflows;
3. use momentum-like variables designed to remove self-dependence;
4. treat the geometric acceleration explicitly, accepting an additional approximation.

This is an important difference from the slab algorithm: the complete toroidal geometric operator cannot, in general, be represented only by the original constant-shift interpolations.

---

## 9. Symmetric time splitting

Write the transformed equation schematically as

$
\partial_t g
=
(\mathcal L_D+\mathcal L_E+\mathcal L_{\delta B}
+\mathcal L_G+\mathcal L_S)g.
$

A second-order step can be organized as

$
g^{n+1}
=
\mathcal S_{\mathrm{src}}(\Delta t/2)
\mathcal K(\Delta t/2)
\mathcal D(\Delta t)
\mathcal K(\Delta t/2)
\mathcal S_{\mathrm{src}}(\Delta t/2)
g^n,
$

where

$
\mathcal K(h)
=
\mathcal E(h/2)
\mathcal B_\delta(h/2)
\mathcal G(h)
\mathcal B_\delta(h/2)
\mathcal E(h/2).
$

Here:

- $\mathcal D$ contains the three spatial interpolations;
- $\mathcal E$ contains the electric logical-velocity translations;
- $\mathcal B_\delta$ contains only the residual field $\delta\mathbf B$;
- $\mathcal G$ contains the geometric curvature terms;
- the dominant $\mathbf B_0$ Lorentz rotation is already represented analytically by $\mathsf Q(t)$.

The electric field should be evaluated at the midpoint. A practical procedure is:

1. compute $n_i^n$, $\phi^n$, and $\mathbf E^n$;
2. predict $g^{n+1/2,*}$;
3. compute $\mathbf E^{n+1/2,*}$ from the predicted density;
4. perform the full symmetric step using this midpoint field;
5. optionally iterate the midpoint field.

The absolute time must be passed to every suboperator because $\mathsf Q(t)$ is explicitly time dependent.

---

## 10. Boundary conditions

### Periodic $y$ and $z$ boundaries

Use

$
g(x,y+L_y,z,\mathbf u)
=
g(x,y,z,\mathbf u),
$

$
g(x,y,z+L_z,\mathbf u)
=
g(x,y,z,\mathbf u).
$

Departure points are wrapped modulo $L_y$ or $L_z$.

These are local numerical periodicities. In particular, periodicity in the short $y$-domain does not represent one complete poloidal transit.

### Specular radial reflection

Specular reflection is defined in physical velocity:

$
v_x\rightarrow-v_x,
\qquad
v_y\rightarrow v_y,
\qquad
v_z\rightarrow v_z.
$

With

$
\mathsf S_x=\operatorname{diag}(-1,1,1),
$

the physical reflection is

$
\mathbf v'\!=\mathsf S_x\mathbf v.
$

In logical velocity coordinates,

$
\mathbf u'
=
\mathsf Q^T(t)\mathsf S_x\mathsf Q(t)\mathbf u.
$

For an $x$-advection, reflect the spatial departure point:

$
x_d\mapsto2x_{\min}-x_d
$

at the inner wall, or

$
x_d\mapsto2x_{\max}-x_d
$

at the outer wall, and sample the distribution at the transformed logical velocity $\mathbf u'$.

Unlike the slab case with a grid-aligned radial velocity, this reflection generally mixes $u_x$ and $u_y$. It therefore requires:

- a multidimensional logical-velocity interpolation;
- a decomposition into one-dimensional shears; or
- a special boundary representation using physical radial velocity.

It is not generally a simple permutation between $u_x$ and $-u_x$.

An even electrostatic extension is compatible with

$
E_x(x_{\min})=E_x(x_{\max})=0.
$

---

## 11. Physics retained

The model contains:

- full ion gyromotion;
- finite-Larmor-radius ion dynamics;
- the dominant Lorentz force through the logical rotation matrix;
- poloidal and toroidal magnetic-field components;
- safety-factor-dependent magnetic pitch;
- optional local magnetic shear;
- toroidal field-line curvature;
- grad-$B$ effects;
- radial ion-density and ion-temperature gradients;
- ion parallel dynamics within the local box;
- adiabatic-electron quasineutrality;
- nonlinear electrostatic transport;
- zonal-flow generation;
- local bad-curvature ITG and interchange-type instability.

At the outboard midplane, the radial temperature gradient and unfavorable curvature provide the free energy and geometric coupling needed for locally ballooning toroidal ITG modes.

---

## 12. Physics not retained

The short frozen-poloidal patch does not contain:

- the complete ballooning eigenfunction around a flux surface;
- favorable-curvature regions;
- full parallel connection over $2\pi qR_0$;
- toroidal trapped-particle trajectories;
- mirror bounce motion;
- the global trapped-passing boundary;
- global profile and equilibrium evolution;
- global twist-and-shift structure unless added separately.

The model is therefore a local, full-orbit, bad-curvature approximation rather than a global toroidal simulation.

---

## 13. Main differences from the implemented slab model

| Feature | Slab model | Local toroidal model |
|---|---|---|
| Geometry | Cartesian and homogeneous | Local patch of a curved tokamak flux surface |
| Reference field | $\mathbf B_0=\hat{\mathbf z}$ | Same reference field, plus $B_y$, radial variation, and curvature |
| Logical rotating grid | Removes the complete uniform Lorentz force | Removes only the chosen constant reference-field Lorentz force |
| Residual magnetic force | Usually zero | Contains $(\mathsf Q\mathbf u)\times\delta\mathbf B$ |
| Spatial coefficients | $\mathsf Q\mathbf u$ | Same structure, with geometry-dependent field coefficients |
| Geometric acceleration | Absent | Contains toroidal and optional poloidal curvature terms |
| Safety factor | Absent | Sets $B_y/B_z$ and local field-line pitch |
| Magnetic shear | Absent | Enters through $q(x)$ |
| Grad-$B$ drift | Absent for homogeneous $B$ | Produced by $B_z(x)$ |
| ITG curvature drive | Absent | Present on the outboard bad-curvature side |
| Particle trapping | Absent | Still absent because the short patch freezes $B(\theta)$ |
| Velocity interpolation | Constant translations and simple shears | Additional variable-coefficient geometric subflows are required |
| Radial reflection | Often a grid permutation | Generally mixes logical velocity components |
| Periodic $y$ boundary | Physical slab periodicity | Artificial tiling of a short poloidal patch |

The central implementation change is therefore not the logical rotating grid itself: that part remains unchanged. The new work consists of adding the residual magnetic field, safety-factor-dependent pitch, geometric acceleration, radial field-strength variation, and the more complicated radial reflection and geometric velocity-space subflows.
