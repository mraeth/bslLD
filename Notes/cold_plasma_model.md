Yes. Below is the **species susceptibility form** of the cold magnetized plasma response, normalized to the **ion cyclotron frequency** so that the **electron–ion mass ratio appears explicitly**.

I will use a singly ionized electron–ion plasma unless stated otherwise.

---

# Cold Plasma Susceptibility Form Normalized to Ion Cyclotron Frequency

Assume

$$
\mathbf{B}_0 = B_0 \hat{\mathbf{z}}
$$

and plane-wave perturbations

$$
\propto \exp\left[i\left(\mathbf{k}\cdot \mathbf{r}-\omega t\right)\right].
$$

The dielectric tensor is written as

$$
\boldsymbol{\varepsilon}
=
\mathbf{I}
+
\sum_s \boldsymbol{\chi}_s ,
$$

where \(\boldsymbol{\chi}_s\) is the susceptibility tensor of species \(s\).

---

## 1. Normalized Variables

Normalize all frequencies to the ion cyclotron frequency

$$
\Omega_i = \frac{eB_0}{m_i}.
$$

Define

$$
\bar{\omega}
=
\frac{\omega}{\Omega_i},
$$

and for each species

$$
\bar{\Omega}_s
=
\frac{\Omega_s}{\Omega_i}.
$$

For ions,

$$
\bar{\Omega}_i = 1.
$$

For electrons,

$$
\Omega_e = -\frac{eB_0}{m_e},
$$

so

$$
\bar{\Omega}_e
=
\frac{\Omega_e}{\Omega_i}
=
-\frac{m_i}{m_e}.
$$

Define the ion–electron mass ratio

$$
\mu = \frac{m_i}{m_e}.
$$

Therefore,

$$
\bar{\Omega}_e = -\mu.
$$

For protons,

$$
\mu \approx 1836.
$$

---

## 2. Normalized Plasma Frequencies

Define

$$
\alpha_s
=
\frac{\omega_{ps}^2}{\Omega_i^2}.
$$

For ions,

$$
\alpha_i
=
\frac{\omega_{pi}^2}{\Omega_i^2}.
$$

For electrons,

$$
\alpha_e
=
\frac{\omega_{pe}^2}{\Omega_i^2}.
$$

For a quasineutral singly ionized plasma,

$$
n_e = n_i,
$$

and

$$
\frac{\omega_{pe}^2}{\omega_{pi}^2}
=
\frac{m_i}{m_e}
=
\mu.
$$

Hence

$$
\alpha_e = \mu \alpha_i.
$$

It is convenient to define

$$
\alpha \equiv \alpha_i
=
\frac{\omega_{pi}^2}{\Omega_i^2}.
$$

Then

$$
\alpha_e = \mu \alpha.
$$

---

# 3. Species Susceptibility Tensor

For each species \(s\), the cold-plasma susceptibility tensor is

$$
\boldsymbol{\chi}_s
=
\begin{pmatrix}
\chi_{s,\perp} & -i\chi_{s,H} & 0 \\
i\chi_{s,H} & \chi_{s,\perp} & 0 \\
0 & 0 & \chi_{s,\parallel}
\end{pmatrix},
$$

where

$$
\chi_{s,\perp}
=
-
\frac{\alpha_s}{\bar{\omega}^2-\bar{\Omega}_s^2},
$$

$$
\chi_{s,H}
=
\frac{\bar{\Omega}_s}{\bar{\omega}}
\frac{\alpha_s}{\bar{\omega}^2-\bar{\Omega}_s^2},
$$

and

$$
\chi_{s,\parallel}
=
-
\frac{\alpha_s}{\bar{\omega}^2}.
$$

Thus,

$$
\boldsymbol{\varepsilon}
=
\mathbf{I}
+
\sum_s
\boldsymbol{\chi}_s .
$$

---

# 4. Ion Susceptibility

For the ions,

$$
\bar{\Omega}_i = 1,
$$

and

$$
\alpha_i = \alpha.
$$

Therefore,

$$
\chi_{i,\perp}
=
-
\frac{\alpha}{\bar{\omega}^2-1},
$$

$$
\chi_{i,H}
=
\frac{1}{\bar{\omega}}
\frac{\alpha}{\bar{\omega}^2-1},
$$

$$
\chi_{i,\parallel}
=
-
\frac{\alpha}{\bar{\omega}^2}.
$$

So the ion susceptibility tensor is

$$
\boldsymbol{\chi}_i
=
\begin{pmatrix}
-\dfrac{\alpha}{\bar{\omega}^2-1}
&
-i\dfrac{\alpha}{\bar{\omega}(\bar{\omega}^2-1)}
&
0
\\[1.2em]
i\dfrac{\alpha}{\bar{\omega}(\bar{\omega}^2-1)}
&
-\dfrac{\alpha}{\bar{\omega}^2-1}
&
0
\\[1.2em]
0 & 0 &
-\dfrac{\alpha}{\bar{\omega}^2}
\end{pmatrix}.
$$

---

# 5. Electron Susceptibility with Explicit Mass Ratio

For electrons,

$$
\bar{\Omega}_e = -\mu,
$$

and

$$
\alpha_e = \mu \alpha.
$$

Therefore,

$$
\chi_{e,\perp}
=
-
\frac{\mu\alpha}{\bar{\omega}^2-\mu^2},
$$

$$
\chi_{e,H}
=
\frac{-\mu}{\bar{\omega}}
\frac{\mu\alpha}{\bar{\omega}^2-\mu^2},
$$

or

$$
\chi_{e,H}
=
-
\frac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)}.
$$

The parallel electron susceptibility is

$$
\chi_{e,\parallel}
=
-
\frac{\mu\alpha}{\bar{\omega}^2}.
$$

Thus,

$$
\boldsymbol{\chi}_e
=
\begin{pmatrix}
-\dfrac{\mu\alpha}{\bar{\omega}^2-\mu^2}
&
i\dfrac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)}
&
0
\\[1.2em]
-i\dfrac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)}
&
-\dfrac{\mu\alpha}{\bar{\omega}^2-\mu^2}
&
0
\\[1.2em]
0 & 0 &
-\dfrac{\mu\alpha}{\bar{\omega}^2}
\end{pmatrix}.
$$

The electron mass ratio appears explicitly through

$$
\mu = \frac{m_i}{m_e}.
$$

---

# 6. Total Cold-Plasma Dielectric Tensor

The total susceptibility is

$$
\boldsymbol{\chi}
=
\boldsymbol{\chi}_i
+
\boldsymbol{\chi}_e .
$$

Therefore,

$$
\boldsymbol{\varepsilon}
=
\mathbf{I}
+
\boldsymbol{\chi}_i
+
\boldsymbol{\chi}_e .
$$

The dielectric tensor can still be written as

$$
\boldsymbol{\varepsilon}
=
\begin{pmatrix}
S & -iD & 0 \\
iD & S & 0 \\
0 & 0 & P
\end{pmatrix}.
$$

In terms of the susceptibilities,

$$
S = 1 + \chi_{i,\perp} + \chi_{e,\perp},
$$

$$
D = \chi_{i,H} + \chi_{e,H},
$$

$$
P = 1 + \chi_{i,\parallel} + \chi_{e,\parallel}.
$$

Explicitly,

$$
S
=
1
-
\frac{\alpha}{\bar{\omega}^2-1}
-
\frac{\mu\alpha}{\bar{\omega}^2-\mu^2},
$$

$$
D
=
\frac{\alpha}{\bar{\omega}(\bar{\omega}^2-1)}
-
\frac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)},
$$

and

$$
P
=
1
-
\frac{\alpha}{\bar{\omega}^2}
-
\frac{\mu\alpha}{\bar{\omega}^2}.
$$

Thus,

$$
P
=
1
-
\frac{(1+\mu)\alpha}{\bar{\omega}^2}.
$$

---

# 7. Right- and Left-Hand Susceptibilities

It is often useful to work in the circular polarization basis. Define

$$
R = S + D,
$$

$$
L = S - D.
$$

Each species contributes

$$
R = 1 + \sum_s \chi_{R,s},
$$

$$
L = 1 + \sum_s \chi_{L,s}.
$$

The species susceptibilities are

$$
\chi_{R,s}
=
-
\frac{\alpha_s}{\bar{\omega}\left(\bar{\omega}+\bar{\Omega}_s\right)},
$$

and

$$
\chi_{L,s}
=
-
\frac{\alpha_s}{\bar{\omega}\left(\bar{\omega}-\bar{\Omega}_s\right)}.
$$

---

## Ion Contributions

For ions,

$$
\bar{\Omega}_i = 1,
$$

so

$$
\chi_{R,i}
=
-
\frac{\alpha}{\bar{\omega}(\bar{\omega}+1)},
$$

and

$$
\chi_{L,i}
=
-
\frac{\alpha}{\bar{\omega}(\bar{\omega}-1)}.
$$

---

## Electron Contributions

For electrons,

$$
\bar{\Omega}_e = -\mu,
$$

and

$$
\alpha_e = \mu\alpha.
$$

Therefore,

$$
\chi_{R,e}
=
-
\frac{\mu\alpha}{\bar{\omega}(\bar{\omega}-\mu)},
$$

and

$$
\chi_{L,e}
=
-
\frac{\mu\alpha}{\bar{\omega}(\bar{\omega}+\mu)}.
$$

So

$$
R
=
1
-
\frac{\alpha}{\bar{\omega}(\bar{\omega}+1)}
-
\frac{\mu\alpha}{\bar{\omega}(\bar{\omega}-\mu)},
$$

and

$$
L
=
1
-
\frac{\alpha}{\bar{\omega}(\bar{\omega}-1)}
-
\frac{\mu\alpha}{\bar{\omega}(\bar{\omega}+\mu)}.
$$

The electron cyclotron resonance appears at

$$
\bar{\omega} = \mu,
$$

because

$$
|\Omega_e| = \mu \Omega_i.
$$

---

# 8. Strong Magnetic Field / Low-Frequency Limit

For frequencies well below the electron cyclotron frequency,

$$
\bar{\omega} \ll \mu,
$$

the electron perpendicular susceptibility becomes

$$
\chi_{e,\perp}
=
-
\frac{\mu\alpha}{\bar{\omega}^2-\mu^2}
\approx
\frac{\alpha}{\mu}.
$$

The electron Hall susceptibility becomes

$$
\chi_{e,H}
=
-
\frac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)}
\approx
\frac{\alpha}{\bar{\omega}}.
$$

For ions, if also

$$
\bar{\omega} \ll 1,
$$

then

$$
\chi_{i,\perp}
\approx
\alpha,
$$

and

$$
\chi_{i,H}
\approx
-\frac{\alpha}{\bar{\omega}}.
$$

Therefore, at very low frequency,

$$
\chi_{i,H}+\chi_{e,H}
\approx
-\frac{\alpha}{\bar{\omega}}
+
\frac{\alpha}{\bar{\omega}}
\approx 0.
$$

This cancellation reflects the fact that, at sufficiently low frequencies, ions and electrons undergo nearly the same \(\mathbf{E}\times\mathbf{B}\) drift.

---

# Final Compact Form

With

$$
\bar{\omega} = \frac{\omega}{\Omega_i},
\qquad
\mu = \frac{m_i}{m_e},
\qquad
\alpha = \frac{\omega_{pi}^2}{\Omega_i^2},
$$

the ion susceptibility is

$$
\boldsymbol{\chi}_i
=
\begin{pmatrix}
-\dfrac{\alpha}{\bar{\omega}^2-1}
&
-i\dfrac{\alpha}{\bar{\omega}(\bar{\omega}^2-1)}
&
0
\\[1.2em]
i\dfrac{\alpha}{\bar{\omega}(\bar{\omega}^2-1)}
&
-\dfrac{\alpha}{\bar{\omega}^2-1}
&
0
\\[1.2em]
0 & 0 &
-\dfrac{\alpha}{\bar{\omega}^2}
\end{pmatrix},
$$

and the electron susceptibility is

$$
\boldsymbol{\chi}_e
=
\begin{pmatrix}
-\dfrac{\mu\alpha}{\bar{\omega}^2-\mu^2}
&
i\dfrac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)}
&
0
\\[1.2em]
-i\dfrac{\mu^2\alpha}{\bar{\omega}(\bar{\omega}^2-\mu^2)}
&
-\dfrac{\mu\alpha}{\bar{\omega}^2-\mu^2}
&
0
\\[1.2em]
0 & 0 &
-\dfrac{\mu\alpha}{\bar{\omega}^2}
\end{pmatrix}.
$$

The full cold-plasma dielectric tensor is

$$
\boldsymbol{\varepsilon}
=
\mathbf{I}
+
\boldsymbol{\chi}_i
+
\boldsymbol{\chi}_e .
$$