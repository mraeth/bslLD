# Memo: Solving Maxwell's Equations in a Vacuum Using `bslLD`

**Author:** [Your Name]  
**Date:** 2026-05-18  
**Subject:** Numerical solution of Maxwell's equations in vacuum with the `bslLD` code

---

## 1. Overview

This memo describes how to use the `bslLD` code to solve Maxwell's equations in a vacuum. The goal is to [briefly state goal, e.g., simulate electromagnetic wave propagation / compute field evolution from given initial conditions].

---

## 2. Governing Equations

Maxwell's equations in a vacuum (SI units, no free charges or currents) are:

$$\nabla \cdot \mathbf{E} = 0$$

$$\nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \times \mathbf{B} = \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

where $\mathbf{E}$ is the electric field, $\mathbf{B}$ is the magnetic field, $\varepsilon_0$ is the permittivity of free space, and $\mu_0$ is the permeability of free space. Together these imply wave propagation at speed $c = 1/\sqrt{\mu_0 \varepsilon_0}$.

---

## 3. Numerical Approach in `bslLD`

### 3.1 Spatial Discretization

Spatial derivatives are computed **spectrally**: each field component is transformed to Fourier space via FFT, multiplied by $ik_d$ (the wavenumber in direction $d$), and transformed back. This gives spectral accuracy on periodic domains and is described in detail in Section 5.

### 3.2 Time Stepping

Time integration uses the **Crank–Nicolson (CN)** scheme — a second-order, unconditionally stable implicit method. Because the Maxwell curl equations are linear, CN reduces to an explicit update that can be evaluated directly without solving a linear system at each step. The scheme is described in detail in Section 6.

### 3.3 Boundary Conditions

The spectral discretization assumes **periodic boundary conditions** in all spatial directions.

---

## 4. Field Representation

The electromagnetic fields $\mathbf{E}$ and $\mathbf{B}$ are represented using two parametric Julia structs that wrap backend-allocated arrays. This design abstracts over storage backends (CPU, GPU, etc.) while enforcing dimensional and type consistency at construction time.

### 4.1 `ScalarField`

A `ScalarField` holds a single $N$-dimensional array of element type `DT`, allocated on the active backend via `bslLD.backend_array`:

```julia
struct ScalarField{DT, N, AT<:AbstractArray{DT,N}}
    data :: AT

    function ScalarField(data::AbstractArray{DT,N}) where {DT,N}
        allocated = bslLD.backend_array(data)
        return new{eltype(allocated), ndims(allocated), typeof(allocated)}(allocated)
    end
end
```

The three type parameters are:

| Parameter | Meaning |
|-----------|---------|
| `DT` | Element type (e.g. `Float64`) |
| `N` | Number of array dimensions |
| `AT` | Concrete array type after backend allocation |

Passing through `bslLD.backend_array` at construction ensures the stored array lives on the intended device; the type parameter `AT` then carries that information statically.

### 4.2 `VectorField`

A `VectorField` bundles an ordered `Vector` of `ScalarField` components. It represents a vector-valued field such as $\mathbf{E} = (E_x, E_y, E_z)$:

```julia
struct VectorField{DT, N, SF<:ScalarField{DT,N}}
    data :: Vector{SF}

    function VectorField(data::AbstractVector{<:ScalarField})
        isempty(data) && throw(ArgumentError("VectorField requires at least one component"))
        first_component = first(data)
        component_axes  = axes(first_component.data)
        component_type  = typeof(first_component)

        for component in data
            axes(component.data) == component_axes ||
                throw(DimensionMismatch("VectorField component axes must match"))
            typeof(component) == component_type ||
                throw(ArgumentError("VectorField components must have the same concrete array type after allocation"))
        end
        return new{eltype(first_component.data), ndims(first_component.data), component_type}(collect(data))
    end
end

# Convenience constructor: wraps raw arrays into ScalarFields automatically
function VectorField(data::AbstractVector{<:AbstractArray})
    return VectorField(ScalarField.(data))
end
```

The inner constructor enforces two invariants at construction time:

- **Axis compatibility** — all components must share the same `axes`, preventing silent shape mismatches.
- **Type homogeneity** — all components must have the same concrete `ScalarField` type, which ensures a uniform backend and element type across the vector.

The `Base` interface methods make `VectorField` behave like a standard collection:

```julia
getindex(field::VectorField, i::Int) = field.data[i]   # field[1] → Ex, field[2] → Ey, …
length(field::VectorField)           = length(field.data)
iterate(field::VectorField, state…)  = iterate(field.data, state…)
```

### 4.3 Constructing the Maxwell Fields

With the above types, $\mathbf{E}$ and $\mathbf{B}$ are constructed as:

```julia
# 3D grid of size nx × ny × nz, double precision
Ex = ScalarField(zeros(nx, ny, nz))
Ey = ScalarField(zeros(nx, ny, nz))
Ez = ScalarField(zeros(nx, ny, nz))

E = VectorField([Ex, Ey, Ez])
B = VectorField([ScalarField(zeros(nx, ny, nz)) for _ in 1:3])

# Access a component
E[1]          # Ex as a ScalarField
E[1].data     # the underlying array
```

---

## 5. Spectral Differential Operators

All spatial derivatives are evaluated spectrally. A derivative in direction $d$ of a field $f$ is computed as:

$$\frac{\partial f}{\partial x_d} = \mathcal{F}^{-1}\!\left[ ik_d \,\hat{f} \right]$$

where $\hat{f} = \mathcal{F}[f]$ is the FFT of $f$ along direction $d$ and $k_d$ are the standard FFT wavenumbers.

### 5.1 Wavenumber Array

`spectral_wavenumbers` builds the wavenumber array for direction `dir` from the grid spacing and grid size:

```julia
function spectral_wavenumbers(field_data, grid, dir)
    n           = size(field_data, dir)
    axis_length = n * grid.delta[dir]
    k = similar(field_data, Float64, n)
    copyto!(k, collect((2π / axis_length) .* fftfreq(n, n)))
    return k
end
```

### 5.2 `differentiate`

`differentiate` applies $ik_d$ in Fourier space using the GPU-/CPU-portable `differentiate_kernel!`:

```julia
function differentiate(field::ScalarField, grid::Grid, dir::Int)
    fhat = fft(field.data, dir)           # forward FFT along dir
    k    = spectral_wavenumbers(field.data, grid, dir)
    ctx  = DifferentiateContext(k, size(fhat), dir)
    kernel!(fhat, ctx; ndrange=length(fhat))  # fhat[i] *= im * k[i_dir]
    KernelAbstractions.synchronize(exec)
    return ScalarField(real(ifft(fhat, dir)))  # inverse FFT, take real part
end
```

The `DifferentiateContext` maps a flat global index back to an $N$-dimensional index via `index_1d_to_nd`, then reads $k_{i_d}$:

```julia
@inline function compute_derivative_multiplier(ctx, index)
    idxs = index_1d_to_nd(index, ctx.sizes)
    return im * ctx.k[idxs[ctx.dir]]
end
```

### 5.3 `grad`, `div`, `curl`

Higher-order operators are built from `differentiate`:

**Gradient** of a scalar field $\phi$:

$$\nabla \phi = \left(\frac{\partial \phi}{\partial x_1},\, \ldots,\, \frac{\partial \phi}{\partial x_d}\right)$$

```julia
grad(field::ScalarField, grid) = VectorField([differentiate(field, grid, d) for d in 1:spatial_ndims(grid)])
```

**Divergence** of a vector field $\mathbf{F}$:

$$\nabla \cdot \mathbf{F} = \sum_{d=1}^{D} \frac{\partial F_d}{\partial x_d}$$

```julia
# implemented as a running sum of differentiate(field[d], grid, d)
```

**Curl** in 3D (used directly for Maxwell):

$$\nabla \times \mathbf{F} = \begin{pmatrix} \partial_y F_z - \partial_z F_y \\ \partial_z F_x - \partial_x F_z \\ \partial_x F_y - \partial_y F_x \end{pmatrix}$$

The `curl` function handles 1D, 2D (returning a scalar or 3-component field depending on the number of components), and 3D cases with appropriate argument checks.

---

## 6. Crank–Nicolson Time Integration

### 6.1 System Structure in Fourier Space

After taking the spatial DFT, each wavenumber mode $\mathbf{k}$ decouples from all others. The curl operator becomes $\nabla\times \leftrightarrow C_\mathbf{k}(\cdot) := i\mathbf{k}\times(\cdot)$, and the Maxwell curl equations reduce to a coupled ODE system per mode:

$$\frac{d}{dt}\begin{pmatrix}\hat{\mathbf{E}}\\ \hat{\mathbf{B}}\end{pmatrix} = \underbrace{\begin{pmatrix}0 & c^2 C_\mathbf{k}\\ -C_\mathbf{k} & 0\end{pmatrix}}_{L_\mathbf{k}} \begin{pmatrix}\hat{\mathbf{E}}\\ \hat{\mathbf{B}}\end{pmatrix}$$

The matrix $L_\mathbf{k}$ is **skew-Hermitian**: $L_\mathbf{k}^\dagger = -L_\mathbf{k}$. Its eigenvalues are $\pm ic|\mathbf{k}|$ — purely imaginary — which is the algebraic signature of energy conservation in the continuous problem.

### 6.2 Why the Implicit System Has a Closed-Form Inverse

The CN discretisation of $\dot{u} = L_\mathbf{k} u$ is:

$$\left(I - \frac{\Delta t}{2}L_\mathbf{k}\right)u^{n+1} = \left(I + \frac{\Delta t}{2}L_\mathbf{k}\right)u^n$$

Inverting the left-hand side generally requires a linear solve, but here $L_\mathbf{k}$ satisfies a very special identity. For divergence-free fields, $C_\mathbf{k}^2(\mathbf{v}) = (i\mathbf{k}\times)(i\mathbf{k}\times\mathbf{v}) = -|\mathbf{k}|^2\mathbf{v}$, so:

$$L_\mathbf{k}^2 = -c^2|\mathbf{k}|^2\, I$$

Setting $\alpha = c|\mathbf{k}|\Delta t/2$, this means $(\frac{\Delta t}{2}L_\mathbf{k})^2 = -\alpha^2 I$, so the Neumann series for the inverse terminates after the second term:

$$\left(I - \frac{\Delta t}{2}L_\mathbf{k}\right)^{-1} = \frac{1}{1+\alpha^2}\left(I + \frac{\Delta t}{2}L_\mathbf{k}\right)$$

This can be verified directly: $(I - \frac{\Delta t}{2}L_\mathbf{k})(I + \frac{\Delta t}{2}L_\mathbf{k}) = I - (\frac{\Delta t}{2})^2 L_\mathbf{k}^2 = (1+\alpha^2)I$. The update operator is therefore:

$$R_\mathbf{k} := \left(I - \frac{\Delta t}{2}L_\mathbf{k}\right)^{-1}\left(I + \frac{\Delta t}{2}L_\mathbf{k}\right) = \frac{1}{1+\alpha^2}\left(I + \frac{\Delta t}{2}L_\mathbf{k}\right)^2 = \frac{1}{1+\alpha^2}\left[(1-\alpha^2)I + \Delta t\, L_\mathbf{k}\right]$$

This is the **Cayley transform** of $\frac{\Delta t}{2}L_\mathbf{k}$, and it is a closed-form matrix — no iterative solver needed. Written out for each field:

$$\hat{\mathbf{E}}^{n+1} = \frac{1}{1+\alpha^2}\left[(1-\alpha^2)\hat{\mathbf{E}}^{n} + c^2\Delta t\, (i\mathbf{k}\times\hat{\mathbf{B}}^{n})\right]$$

$$\hat{\mathbf{B}}^{n+1} = \frac{1}{1+\alpha^2}\left[(1-\alpha^2)\hat{\mathbf{B}}^{n} - \Delta t\, (i\mathbf{k}\times\hat{\mathbf{E}}^{n})\right]$$

### 6.3 Why Stability Is Unconditional

The update operator $R_\mathbf{k}$ is the Cayley transform of a skew-Hermitian matrix. The Cayley transform maps skew-Hermitian operators to **unitary** operators — this is a classical result. Unitarity means $R_\mathbf{k}^\dagger R_\mathbf{k} = I$, so all eigenvalues of $R_\mathbf{k}$ lie on the unit circle with $|\lambda| = 1$ exactly, for every $\mathbf{k}$ and every $\Delta t$.

Geometrically (see diagram above): the eigenvalues of $L_\mathbf{k}$ live on the imaginary axis, which is the neutral stability boundary of the continuous system. The Cayley transform is the Möbius map that sends the imaginary axis bijectively onto the unit circle. So every mode that was neutrally stable continuously ends up exactly on the unit circle discretely — no damping, no growth, regardless of $\Delta t$.

### 6.4 Energy Conservation

By Parseval's theorem, the total electromagnetic energy satisfies:

$$\mathcal{H} \propto \sum_\mathbf{k}\left(|\hat{\mathbf{E}}_\mathbf{k}|^2 + c^2|\hat{\mathbf{B}}_\mathbf{k}|^2\right)$$

Since $R_\mathbf{k}$ is unitary, $\|u^{n+1}_\mathbf{k}\| = \|u^n_\mathbf{k}\|$ for every mode. The total discrete energy is therefore conserved **to machine precision** at every time step — a property that explicit schemes like leapfrog or RK4 only approximate up to their truncation error.

### 6.5 Summary of Properties

| Property | Result | Reason |
|----------|--------|--------|
| Order of accuracy | 2nd order in $\Delta t$ | CN is a trapezoidal rule |
| Stability | Unconditionally stable | $R_\mathbf{k}$ is unitary (Cayley transform of skew-Hermitian) |
| No linear solve | Explicit update formula | $L_\mathbf{k}^2 = -\alpha^2 I$ makes inverse closed-form |
| Energy conservation | Exact (machine precision) | Unitary $R_\mathbf{k}$ preserves mode norms exactly |
| Divergence preservation | Exact | $\mathbf{k}\cdot\hat{\mathbf{f}} = 0$ is not modified by $C_\mathbf{k}$ or $R_\mathbf{k}$ |

### 6.6 Implementation Sketch

```julia
function step_cn!(E::VectorField, B::VectorField, grid::Grid, dt::Real, c::Real)
    # Compute spectral curls: ∇×E and ∇×B using Section 5 operators
    curlE = curl(E, grid)
    curlB = curl(B, grid)

    α² = (c * |k| * dt/2)²   # applied per-mode inside the FFT
    prefac = 1 / (1 + α²)

    for d in 1:3
        # closed-form Cayley update per component
        E[d].data .= prefac .* ((1 - α²) .* E[d].data .+ c^2 * dt .* curlB[d].data)
        B[d].data .= prefac .* ((1 - α²) .* B[d].data .- dt       .* curlE[d].data)
    end
end
```

> **Note:** In the actual spectral implementation, `α²` is a wavenumber-dependent array $c^2|\mathbf{k}|^2(\Delta t/2)^2$ computed from `spectral_wavenumbers` and broadcast over the Fourier coefficients before the inverse FFT. The per-mode scalar $\alpha^2$ in the sketch above becomes an array multiply in practice.

