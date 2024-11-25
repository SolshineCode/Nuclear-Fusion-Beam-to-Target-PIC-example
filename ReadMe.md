## Plasma Fusion Simulation Teaching Model

MIT License

# Overview
Simple educational model demonstrating beam-to-target fusion concepts using particle tensors. Models (as a concept, not accurately) deuterium beam impacting tritium target. This is inspired from some recent discussions about my past work with the beam to target fusion reactors and the usefulness of tensor-based plasma simulations in under-studied contexts such as beam-to-target reactors.


# Key Features
* Particle representation using NumPy arrays (position, velocity tensors)
* Coulomb force calculations between particles (mock parameter, representing more complex calculations)
* Extremely Basic fusion probability model
* Time evolution visualization using MatPlotLib


# Usage
```
sim = PlasmaSimulation(n_beam=1000, n_target=1000)
fusion_history = sim.run(steps=1000)
```
Note that the number of steps can be intensive on systems without a GPU (and this program is not CUDA optimized) so you may start with 100 to get a general feel for it.

# Important Considerations

Physics Simplifications
* Classical mechanics only (no quantum effects)
* No magnetic confinement
* Simplified collision model
* No plasma temperature/pressure effects
* No electromagnetic waves

Numerical Considerations
* Timestep sensitivity
* Particle count affects computational load
* Force calculations are O(n²)
* No boundary conditions implemented

Intention is as an Educational Toy Model
* Demonstrates tensor-based particle physics
* Shows basic collision detection
* Illustrates Coulomb interactions
* Good introduction to PIC (Particle-in-cell) methods (a method that has served fruitful in my prior work in nuclear fusion reactors)

Some of the Limitations vs Real Systems
* No energy conservation
* Missing magnetic fields (could add rings, for example, similar to my past work with AGNI reactor)
* No relativistic effects
* Simplified cross sections

# Dependencies
NumPy
Matplotlib
Python 3.x