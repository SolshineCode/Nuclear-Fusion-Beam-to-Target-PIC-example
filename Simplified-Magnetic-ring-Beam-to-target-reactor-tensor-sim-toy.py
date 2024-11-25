import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ReactorParams:
    """Physical parameters for the reactor simulation"""
    tube_length: float = 2.0  # meters
    tube_radius: float = 0.1  # meters
    deuterium_mass: float = 3.34e-27  # kg
    deuterium_charge: float = 1.6e-19  # Coulombs
    lens_positions: List[float] = None  # positions of magnetic lenses
    lens_strengths: List[float] = None  # magnetic field strengths
    target_position: float = 1.9  # meters from start
    
class MagneticLens:
    """Represents a magnetic lens in the reactor"""
    def __init__(self, position: float, strength: float, aperture: float):
        self.position = position
        self.strength = strength
        self.aperture = aperture
    
    def get_field(self, x: float, r: float) -> Tuple[float, float]:
        """Calculate magnetic field components (Bx, Br) at given position"""
        dx = x - self.position
        field_strength = self.strength * np.exp(-(dx**2 + r**2)/(2*self.aperture**2))
        Bx = field_strength * dx/self.aperture
        Br = field_strength * r/self.aperture
        return Bx, Br

class PlasmaSimulation:
    def __init__(self):
        self.params = ReactorParams()
        
        # Initialize magnetic lenses
        self.lenses = [
            MagneticLens(0.5, 1.0, 0.05),
            MagneticLens(1.0, 1.2, 0.05),
            MagneticLens(1.5, 1.5, 0.05)
        ]
        
        # Initialize beam particles
        self.n_particles = 1000
        self.beam_positions = np.zeros((self.n_particles, 3))  # x,y,z coordinates
        self.beam_velocities = np.zeros((self.n_particles, 3))
        
        # Set initial conditions
        self.beam_positions[:,0] = 0.0  # start at x=0
        self.beam_velocities[:,0] = 1e5  # initial velocity in x direction
        
        # Random initial radial positions
        r = np.random.uniform(0, 0.02, self.n_particles)
        theta = np.random.uniform(0, 2*np.pi, self.n_particles)
        self.beam_positions[:,1] = r * np.cos(theta)
        self.beam_positions[:,2] = r * np.sin(theta)
        
        self.fusion_events = []

    def get_magnetic_force(self) -> np.ndarray:
        """Calculate magnetic forces on particles"""
        forces = np.zeros_like(self.beam_positions)
        
        for lens in self.lenses:
            for i in range(self.n_particles):
                x = self.beam_positions[i,0]
                r = np.sqrt(self.beam_positions[i,1]**2 + self.beam_positions[i,2]**2)
                Bx, Br = lens.get_field(x, r)
                
                # F = q(v × B)
                v = self.beam_velocities[i]
                B = np.array([Bx, Br*self.beam_positions[i,1]/r, Br*self.beam_positions[i,2]/r])
                F = self.params.deuterium_charge * np.cross(v, B)
                forces[i] += F
                
        return forces

    def check_fusion(self) -> int:
        """Check for fusion events when particles hit target"""
        fusion_count = 0
        for i in range(self.n_particles):
            if self.beam_positions[i,0] >= self.params.target_position:
                fusion_count += 1
                # Reset particle to start
                self.beam_positions[i,0] = 0.0
                r = np.random.uniform(0, 0.02)
                theta = np.random.uniform(0, 2*np.pi)
                self.beam_positions[i,1] = r * np.cos(theta)
                self.beam_positions[i,2] = r * np.sin(theta)
                
        return fusion_count

    def step(self, dt=1e-9):
        """Advance simulation by one timestep"""
        forces = self.get_magnetic_force()
        self.beam_velocities += forces * dt / self.params.deuterium_mass
        self.beam_positions += self.beam_velocities * dt
        
        fusions = self.check_fusion()
        self.fusion_events.append(fusions)
        return fusions

    def run(self, steps=1000):
        """Run simulation for specified number of steps"""
        for _ in range(steps):
            self.step()
        return np.array(self.fusion_events)

# Run simulation and plot results
sim = PlasmaSimulation()
fusion_history = sim.run()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(fusion_history, label='Fusion events')
plt.xlabel('Time Step')
plt.ylabel('Fusion Events')
plt.title('Fusion Events over Time')
plt.legend()

# Plot final particle positions
plt.subplot(122)
plt.scatter(sim.beam_positions[:,0], 
           np.sqrt(sim.beam_positions[:,1]**2 + sim.beam_positions[:,2]**2),
           alpha=0.1)
plt.xlabel('Position (m)')
plt.ylabel('Radial Distance (m)')
plt.title('Particle Positions')
plt.tight_layout()
plt.show()
