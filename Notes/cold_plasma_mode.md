# Cold-Ion / Drift-Electron Electromagnetic Plasma Model

## Overview

This document describes a hybrid plasma model consisting of:

- Fully kinetic-free cold ion fluid dynamics
- Drift-reduced electrons
- Quasi-neutrality
- Electromagnetic field evolution
- Neglected displacement current

The model is intended for strongly magnetized plasmas satisfying

$$\omega \ll |\Omega_e|,$$

where electron gyromotion is not resolved while ion dynamics remain fully evolved.

---

## Governing Equations

### Assumptions

**Quasi-neutrality:**

$$n_i = n_e = n$$

**Current:**

$$\mathbf{J} = en(\mathbf{u}_i - \mathbf{u}_e)$$

**Neglected displacement current:**

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

**Magnetic field decomposition:**

$$\mathbf{B} = B_0 \hat{z} + \delta\mathbf{B}$$

**Background-field approximation.** The full field $\mathbf{B}$ is evolved via Faraday's law and enters Ampère's law. However, in the Lorentz-force terms and in the $\mathbf{E}\times\mathbf{B}$ drift the field is approximated by the uniform background only:

$$\mathbf{B} \approx B_0\hat{z} \quad \text{(equations of motion and drift only)}.$$

This fixes the parallel unit vector as

$$\mathbf{b} = \hat{z},$$

and simplifies the perpendicular/parallel decomposition throughout.

---

## Ion Equations

### Continuity

$$\frac{\partial n}{\partial t} + \nabla \cdot (n\mathbf{u}_i) = 0$$

### Momentum

With $\mathbf{B} \approx B_0\hat{z}$ in the Lorentz force:

$$m_i \left(\frac{\partial}{\partial t} + \mathbf{u}_i \cdot \nabla\right) \mathbf{u}_i = e\left(\mathbf{E} + \mathbf{u}_i \times B_0\hat{z}\right)$$

---

## Drift-Reduced Electron Model

### Perpendicular Velocity

Electron perpendicular force balance with $\mathbf{B} \approx B_0\hat{z}$:

$$0 = -e\left(\mathbf{E} + \mathbf{u}_e \times B_0\hat{z}\right)_\perp$$

Therefore:

$$\mathbf{u}_{e\perp} = \frac{\mathbf{E} \times \hat{z}}{B_0}$$

### Parallel Velocity

With $\mathbf{b} = \hat{z}$, the electron velocity splits as

$$\mathbf{u}_e = \mathbf{u}_{e\perp} + u_{ez}\,\hat{z}.$$

The parallel momentum equation is

$$m_e \left(\frac{\partial}{\partial t} + \mathbf{u}_e \cdot \nabla\right) u_{ez} = -e E_z,$$

where the parallel electric field is simply

$$E_\parallel = E_z.$$

---

## Electromagnetic Closure

### Current

$$\mathbf{J} = en\left(\mathbf{u}_i - \mathbf{u}_e\right)$$

### Ampère's Law

$$\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$$

The parallel ($z$) current is

$$J_z = \frac{(\nabla \times \mathbf{B})_z}{\mu_0},$$

hence

$$u_{ez} = u_{iz} - \frac{J_z}{en}.$$

This relation replaces explicit evolution of the electron parallel momentum equation.

### Faraday's Law

$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}$$

---

## Conservative Form

Define the state vector

$$\mathbf{U} = \begin{bmatrix} n \\ nu_x \\ nu_y \\ nu_z \\ B_x \\ B_y \\ B_z \end{bmatrix}.$$

The evolved variables are density, ion momentum, and magnetic field. Electron quantities are reconstructed algebraically.

---

## Current Evaluation

Compute the current from Ampère's law spectrally:

$$\hat{\mathbf{J}}_{\mathbf{m}} = \frac{(\widehat{\nabla \times \mathbf{B}})_{\mathbf{m}}}{\mu_0}$$

**Parallel ($z$) current** (using $\mathbf{b} = \hat{z}$):

$$J_z = \mathcal{F}^{-1}\!\left(\frac{i k_x \hat{B}_y - i k_y \hat{B}_x}{\mu_0}\right)$$

**Electron parallel velocity:**

$$u_{ez} = u_{iz} - \frac{J_z}{en}$$

**Full electron velocity** (using $\mathbf{b} = \hat{z}$ and $\mathbf{B} \approx B_0\hat{z}$ in the drift):

$$\mathbf{u}_e = \frac{\mathbf{E} \times \hat{z}}{B_0} + u_{ez}\,\hat{z}$$

---

## Electric Field Recovery

Using $\mathbf{u}_{e\perp} = (\mathbf{E}\times\hat{z})/B_0$ and $\mathbf{B} \approx B_0\hat{z}$:

$$\mathbf{E} = -\mathbf{u}_e \times B_0\hat{z} + E_z\hat{z}$$

The perpendicular components expand to

$$E_x = -B_0\,u_{ey}, \qquad E_y = B_0\,u_{ex}.$$

If electron inertia is neglected, $E_z = 0$.

---

## Time Integration

A second-order Runge–Kutta method is recommended.

**Stage 1:**

$$\mathbf{U}^* = \mathbf{U}^n + \Delta t\, L(\mathbf{U}^n)$$

**Stage 2:**

$$\mathbf{U}^{n+1} = \frac{1}{2}\!\left(\mathbf{U}^n + \mathbf{U}^* + \Delta t\, L(\mathbf{U}^*)\right)$$

where $L(\mathbf{U})$ contains all spatial operators evaluated spectrally.

### Right-Hand Side Terms

**Density update:**

$$\frac{\partial n}{\partial t} = -\nabla \cdot (n\mathbf{u}_i)$$

**Momentum update** (with $\mathbf{B} \approx B_0\hat{z}$ in the Lorentz force):

$$\frac{\partial \mathbf{u}_i}{\partial t} = -(\mathbf{u}_i \cdot \nabla)\mathbf{u}_i + \frac{e}{m_i}\!\left(\mathbf{E} + \mathbf{u}_i \times B_0\hat{z}\right)$$

**Magnetic field update:**

$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}$$

---

## Divergence Constraint

The constraint $\nabla \cdot \mathbf{B} = 0$ is equivalent in Fourier space to

$$\mathbf{k} \cdot \hat{\mathbf{B}}_{\mathbf{m}} = 0 \quad \forall\, \mathbf{m}.$$

Because the spectral Faraday update preserves this identity exactly (up to floating-point rounding) when it is satisfied by the initial condition, no explicit divergence-cleaning step is required. If needed, any accumulated error can be removed by a single projection step in Fourier space:

$$\hat{\mathbf{B}}_{\mathbf{m}} \leftarrow \hat{\mathbf{B}}_{\mathbf{m}} - \frac{\mathbf{k}\left(\mathbf{k}\cdot\hat{\mathbf{B}}_{\mathbf{m}}\right)}{|\mathbf{k}|^2} \quad (\mathbf{k} \neq \mathbf{0}).$$

---

## CFL Condition

The timestep must satisfy

$$\Delta t < C \min\!\left(\frac{\Delta x}{v_{\max}},\; \frac{\Delta y}{v_{\max}},\; \frac{\Delta z}{v_{\max}}\right),$$

where

$$v_{\max} = \max\!\left(|\mathbf{u}_i|,\; v_A\right), \qquad v_A = \frac{B_0}{\sqrt{\mu_0 m_i n}}.$$

Typical values: $C \approx 0.3$–$0.5$.

---

## Algorithm Summary

At each timestep:

1. Compute $\hat{\mathbf{J}} = (\widehat{\nabla \times \mathbf{B}})/\mu_0$ spectrally
2. Extract parallel current $J_z = \mathcal{F}^{-1}(\hat{J}_z)$
3. Recover $u_{ez} = u_{iz} - J_z/(en)$
4. Recover $\mathbf{u}_e = (\mathbf{E}\times\hat{z})/B_0 + u_{ez}\hat{z}$
5. Recover $\mathbf{E} = -\mathbf{u}_e \times B_0\hat{z} + E_z\hat{z}$
6. Advance ion continuity equation (spectral divergence)
7. Advance ion momentum equation (spectral gradient/curl, $\mathbf{B} \approx B_0\hat{z}$)
8. Advance Faraday equation (spectral curl)
9. Apply dealiasing (2/3 rule) to nonlinear terms
10. Proceed to next timestep

This produces a spectrally accurate solver for the cold-ion / drift-electron electromagnetic plasma model under the background-field approximation.