# -*- coding: utf-8 -*-
"""
Atomic scaling module for E2Former.
Adapted from MatterSim's AtomScaling for E2Former architecture.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict
from ase import Atoms


class E2FormerAtomScaling(nn.Module):
    """
    Atomic scaling module for E2Former.
    
    This module provides per-atom-type energy scaling and shifting
    to improve training stability and prediction accuracy.
    """
    
    def __init__(
        self,
        max_z: int = 94,
        scale_key: str = "per_species_energy_std",
        shift_key: str = "per_species_energy_mean", 
        init_scale: Optional[Union[torch.Tensor, float]] = None,
        init_shift: Optional[Union[torch.Tensor, float]] = None,
        trainable_scale: bool = False,
        trainable_shift: bool = False,
        verbose: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Initialize E2FormerAtomScaling.
        
        Args:
            max_z: Maximum atomic number (default: 94)
            scale_key: Scaling strategy key
            shift_key: Shifting strategy key
            init_scale: Initial scaling factors
            init_shift: Initial shifting factors
            trainable_scale: Whether scale parameters are trainable
            trainable_shift: Whether shift parameters are trainable
            verbose: Whether to print initialization info
            device: Device to place tensors on
        """
        super().__init__()
        
        self.max_z = max_z
        self.device = device
        self.verbose = verbose
        
        # Initialize scaling and shifting parameters
        if init_scale is None:
            init_scale = torch.ones(max_z + 1)
        elif isinstance(init_scale, float):
            init_scale = torch.tensor(init_scale).repeat(max_z + 1)
        else:
            assert init_scale.size()[0] == max_z + 1
            
        if init_shift is None:
            init_shift = torch.zeros(max_z + 1)
        elif isinstance(init_shift, float):
            init_shift = torch.tensor(init_shift).repeat(max_z + 1)
        else:
            assert init_shift.size()[0] == max_z + 1
            
        init_shift = init_shift.float()
        init_scale = init_scale.float()
        
        # Register parameters or buffers
        if trainable_scale:
            self.scale = torch.nn.Parameter(init_scale)
        else:
            self.register_buffer("scale", init_scale)
            
        if trainable_shift:
            self.shift = torch.nn.Parameter(init_shift)
        else:
            self.register_buffer("shift", init_shift)
            
        if verbose:
            print(f"E2FormerAtomScaling initialized:")
            print(f"  Scale: {init_scale[:10]}...")  # Show first 10 elements
            print(f"  Shift: {init_shift[:10]}...")
            
        self.to(device)
    
    def compute_statistics_from_data(
        self,
        atoms_list: List[Atoms],
        energies: List[float],
        forces: Optional[List[np.ndarray]] = None,
        method: str = "per_species",
    ):
        """
        Compute scaling statistics from training data.
        
        Args:
            atoms_list: List of ASE Atoms objects
            energies: List of total energies
            forces: List of force arrays (optional)
            method: Statistics computation method
        """
        if method == "per_species":
            self._compute_per_species_statistics(atoms_list, energies)
        elif method == "per_atom":
            self._compute_per_atom_statistics(atoms_list, energies)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_per_species_statistics(
        self, 
        atoms_list: List[Atoms], 
        energies: List[float]
    ):
        """Compute per-species energy statistics."""
        # Collect per-atom energies by species
        species_energies = {z: [] for z in range(self.max_z + 1)}
        
        for atoms, energy in zip(atoms_list, energies):
            per_atom_energy = energy / len(atoms)
            atomic_numbers = atoms.get_atomic_numbers()
            
            for z in atomic_numbers:
                if z <= self.max_z:
                    species_energies[z].append(per_atom_energy)
        
        # Compute mean and std for each species
        new_shift = torch.zeros(self.max_z + 1)
        new_scale = torch.ones(self.max_z + 1)
        
        for z in range(self.max_z + 1):
            if len(species_energies[z]) > 0:
                energies_array = np.array(species_energies[z])
                new_shift[z] = np.mean(energies_array)
                new_scale[z] = np.std(energies_array)
                if new_scale[z] < 1e-6:  # Avoid division by zero
                    new_scale[z] = 1.0
        
        # Update parameters
        with torch.no_grad():
            self.shift.copy_(new_shift)
            self.scale.copy_(new_scale)
        
        if self.verbose:
            print("Updated scaling parameters from data:")
            print(f"  Shift range: [{new_shift.min():.4f}, {new_shift.max():.4f}]")
            print(f"  Scale range: [{new_scale.min():.4f}, {new_scale.max():.4f}]")
    
    def _compute_per_atom_statistics(
        self, 
        atoms_list: List[Atoms], 
        energies: List[float]
    ):
        """Compute global per-atom energy statistics."""
        per_atom_energies = [energy / len(atoms) for atoms, energy in zip(atoms_list, energies)]
        
        mean_energy = np.mean(per_atom_energies)
        std_energy = np.std(per_atom_energies)
        
        # Apply same statistics to all species
        with torch.no_grad():
            self.shift.fill_(mean_energy)
            self.scale.fill_(std_energy)
        
        if self.verbose:
            print(f"Updated scaling parameters (global):")
            print(f"  Shift: {mean_energy:.4f}")
            print(f"  Scale: {std_energy:.4f}")
    
    def transform(
        self, 
        atomic_energies: torch.Tensor, 
        atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform raw atomic energies to normalized values.
        
        Args:
            atomic_energies: Raw atomic energies [N_atoms]
            atomic_numbers: Atomic numbers [N_atoms]
            
        Returns:
            Normalized atomic energies [N_atoms]
        """
        # Clamp atomic numbers to valid range
        atomic_numbers = torch.clamp(atomic_numbers, 0, self.max_z)
        
        # Get scaling parameters for each atom
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        
        # Apply transformation: normalized = scale * raw + shift
        normalized_energies = curr_scale * atomic_energies + curr_shift
        
        return normalized_energies
    
    def inverse_transform(
        self, 
        atomic_energies: torch.Tensor, 
        atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform normalized atomic energies back to raw values.
        
        Args:
            atomic_energies: Normalized atomic energies [N_atoms]
            atomic_numbers: Atomic numbers [N_atoms]
            
        Returns:
            Raw atomic energies [N_atoms]
        """
        # Clamp atomic numbers to valid range
        atomic_numbers = torch.clamp(atomic_numbers, 0, self.max_z)
        
        # Get scaling parameters for each atom
        curr_shift = self.shift[atomic_numbers]
        curr_scale = self.scale[atomic_numbers]
        
        # Apply inverse transformation: raw = (normalized - shift) / scale
        raw_energies = (atomic_energies - curr_shift) / curr_scale
        
        return raw_energies
    
    def forward(
        self, 
        atomic_energies: torch.Tensor, 
        atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass (same as transform).
        
        Args:
            atomic_energies: Raw atomic energies [N_atoms]
            atomic_numbers: Atomic numbers [N_atoms]
            
        Returns:
            Normalized atomic energies [N_atoms]
        """
        return self.transform(atomic_energies, atomic_numbers)
    
    def get_scaling_params(self) -> Dict[str, torch.Tensor]:
        """Get current scaling parameters."""
        return {
            "scale": self.scale.clone(),
            "shift": self.shift.clone()
        }
    
    def set_scaling_params(self, scale: torch.Tensor, shift: torch.Tensor):
        """Set scaling parameters."""
        assert scale.size()[0] == self.max_z + 1
        assert shift.size()[0] == self.max_z + 1
        
        with torch.no_grad():
            self.scale.copy_(scale)
            self.shift.copy_(shift)
    
    def save_scaling_params(self, filepath: str):
        """Save scaling parameters to file."""
        params = self.get_scaling_params()
        torch.save(params, filepath)
        if self.verbose:
            print(f"Scaling parameters saved to {filepath}")
    
    def load_scaling_params(self, filepath: str):
        """Load scaling parameters from file."""
        params = torch.load(filepath, map_location=self.device)
        self.set_scaling_params(params["scale"], params["shift"])
        if self.verbose:
            print(f"Scaling parameters loaded from {filepath}")
