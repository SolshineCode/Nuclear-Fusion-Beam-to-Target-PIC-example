import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PlasmaParameters:
    """Physical constants and parameters"""
    k_coulomb = 8.99e9  # Coulomb constant
    deuterium_mass = 3.34e-27  # kg
    tritium_mass = 5.01e-27    # kg
    particle_charge = 1.60e-19  # C
    fusion_threshold = 0.01     # Arbitrary threshold for fusion probability

class PlasmaSimulation:
    def __init__(self, n_beam=1000, n_target=1000):
        self.params = PlasmaParameters()
        
        # Initialize beam particles (deuterium)
        self.beam_positions = np.random.uniform(-0.01, 0.01, (n_beam, 3))  # m
        self.beam_velocities = np.zeros((n_beam, 3))
        self.beam_velocities[:,0] = np.random.normal(1e5, 1e4, n_beam)  # m/s along x-axis
        
        # Initialize target particles (tritium)
        self.target_positions = np.random.uniform(-0.01, 0.01, (n_target, 3))
        self.target_positions[:,0] += 0.1  # Offset target 10cm downstream
        self.target_velocities = np.zeros((n_target, 3))
        
        self.fusion_events = []

    def calculate_forces(self):
        """Calculate Coulomb forces between particles"""
        forces = np.zeros_like(self.beam_positions)
        
        for i, pos in enumerate(self.beam_positions):
            # Vector from beam particle to all target particles
            r_vectors = self.target_positions - pos
            
            # Calculate distances
            distances = np.linalg.norm(r_vectors, axis=1)
            
            # Calculate Coulomb force magnitude
            force_mag = self.params.k_coulomb * (self.params.particle_charge**2) / (distances**2)
            
            # Calculate force components
            forces[i] = np.sum(force_mag[:,np.newaxis] * r_vectors / distances[:,np.newaxis], axis=0)
            
        return forces

    def check_fusion(self):
        """Check for fusion events based on proximity"""
        fusion_count = 0
        
        for i, beam_pos in enumerate(self.beam_positions):
            distances = np.linalg.norm(self.target_positions - beam_pos, axis=1)
            
            # Simple fusion probability model
            fusion_prob = np.exp(-distances / self.params.fusion_threshold)
            fusion_events = np.random.random(len(distances)) < fusion_prob
            
            fusion_count += np.sum(fusion_events)
            
        return fusion_count

    def step(self, dt=1e-9):
        """Advance simulation by one timestep"""
        forces = self.calculate_forces()
        
        # Update beam particle velocities and positions
        self.beam_velocities += forces * dt / self.params.deuterium_mass
        self.beam_positions += self.beam_velocities * dt
        
        # Record fusion events
        fusions = self.check_fusion()
        self.fusion_events.append(fusions)
        
        return fusions

    def run(self, steps=500):
        """Run simulation for specified number of steps"""
        for _ in range(steps):
            self.step()
        
        return np.array(self.fusion_events)

# Run simulation
sim = PlasmaSimulation()
fusion_history = sim.run()

# Plot results
plt.plot(fusion_history)
plt.xlabel('Time Step')
plt.ylabel('Fusion Events')
plt.title('Fusion Events over Time')
plt.show()