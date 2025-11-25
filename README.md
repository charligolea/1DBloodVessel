# 1DBloodVessel

This repository contains a numerical model for simulating **one-dimensional incompressible blood flow in large arteries**, a problem of both physiological and engineering relevance. The model is based on simplified Navier–Stokes equations — conservation of mass and momentum — coupled with a nonlinear **tube law** relating pressure and vessel area.

The solver implements a **finite volume discretization on a staggered grid** and splits the solution procedure into four stages:

- **Convective stage** (explicit, CFL-limited)  
- **Diffusive stage** (implicit)  
- **Pressure stage** (implicit)  
- **Correction stage**

A first-order time / second-order space integration scheme is used, with spatial fluxes computed using **Kolgan** and **Ducros-type** methods.

Validation was performed through four test cases:  
1. Purely convective flow  
2. Purely diffusive flow  
3. Full-system stationary solution  
4. Full-system unsteady solution  

Across these cases, the numerical results show excellent agreement with analytical reference solutions, with convergence orders matching theoretical predictions.

---

## Repository Structure

```
├── convective.py        # Convective-only simulation
├── diffusive.py         # Diffusive-only simulation
├── stationary.py        # Full-system steady-state simulation
├── unstationary.py      # Full-system unsteady simulation
│
├── src/                 # Core solver implementation
│   ├── BloodVesselDirichlet/  # Dirichlet BC development
│   ├── BloodVesselPeriodic/        # Periodic BCs
│
└── docs/
    └── Simulation of one-dimensional flow in blood vessels with a semi-implicit finite volume scheme.pdf       # Original report with extended theory and results
```

The `src` directory contains the full discretization pipeline, currently supporting **periodic boundary conditions**, with ongoing development toward **Dirichlet boundary conditions**.

---

## Documentation

For detailed derivations, theoretical background, numerical method explanations, and validation results, see the accompanying full report in [docs](https://github.com/charligolea/1DBloodVessel/docs) folder.

This document provides extended insight into the methodology and the reasoning behind the implementation.

---

## Authors

- **Carlos Gómez de Olea Ballester**  
  Technical University of Munich  
  carlos.olea@tum.de  

- **Adrián Juanicotena**  
  Technical University of Munich  
  a.juanicotena@tum.de  

---

## Development Status

This repository is **no longer under active development**.  
It is preserved for reference, reproducibility, and educational purposes.