# 🔥 Heat Diffusion Modelling

A Python implementation of the 2D heat equation to simulate heat diffusion on a surface using finite difference methods.

## Features

### Custom Heat Sources
- `stationary_gaussian()`: Fixed-position Gaussian heat source.
- `lateral_gaussian()`: Horizontally oscillating Gaussian source.
- `circular_gaussian()`: Gaussian source moving in circular motion.
- `uniform()`: Constant heat applied over the entire surface.
- `pulsed()`: Periodic on-off Gaussian heat source.

### Dynamic Material Properties
- Assign thermal conductivity `k`, density `rho`, and heat capacity `c` in any custom region using `set_material_region()`.
- Thermal diffusivity `alpha` is computed automatically as `alpha = k / (rho * c)`.

### 🚧 Boundary Conditions
- Supports **Dirichlet** (fixed temperature) and **Neumann** (zero-gradient) boundary types.
- Easily extendable for additional boundary conditions.

### Realistic Energy Scaling
- Heat sources can be scaled based on physical power input (e.g. sunlight).
- Temperature changes are computed from energy input, material density, and specific heat capacity.

### 2D Animation
- Live animation of the temperature evolution using `matplotlib.animation`.

## Requirements

- Python 3.11
- `numpy`, `matplotlib`, `scipy`

