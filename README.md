# 🔥 Heat diffusion modelling

Implementation of the heat equation to model heat diffusion in a 2D surface.

## Features

Custom heat sources:

- `stationary_gaussian()`: Gaussian heat source which has a fixed position  
- `lateral_gaussian()`: Gaussian heat source which oscillates horizontally  
- `circular_gaussian()`: Gaussian heat source which moves in circular motion  
- `uniform()`: Heat source which is constant over the surface  
- `pulsed()`: Heat sources which pulses every fixed period  

Dynamic material properties:

- User can set thermal conductivity `k`, density `rho`, heat capacity `c` in a customised region using `set_material_region()`  
- Thermal diffusivity calculated automatically  

Various boundary conditions:

- Currently supports Dirichlet (fixed temperature) and Neumann (zero gradient) boundary conditions
- More boundary options to be implemented  

Realistic energy scaling:

- Heat sources can be physically scaled to simulate real world heat sources i.e sunlight  
- Temperature increase is derived from power input and material properties  

2D animation:

- Live animation using `matplotlib`

## Requirements

- Python 3.11  
- `numpy`, `matplotlib`, `scipy`  
