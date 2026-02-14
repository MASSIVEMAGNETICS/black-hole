"""
Gravitational Attention: A Physics-Based Replacement for Transformer Attention

This module implements Gravitational Attention as a drop-in replacement for
standard Transformer attention mechanisms. Instead of computing similarity via
dot products, it simulates gravitational forces between tokens based on their
mass (importance) and distance (semantic gap) in a curved spacetime.

Key Concepts:
- Tokens are "matter" with Position and Mass
- Attention is gravitational force: F = G * (M1 * M2) / (Distance^2 + ε)
- Information orbits around massive concepts
- Hawking Radiation prevents black hole collapse
"""

import numpy as np
from typing import Optional, Tuple
import math


class GravitationalAttentionLayer:
    """
    Gravitational Attention Layer - Physics-based replacement for standard attention.
    
    This layer replaces the Query/Key/Value mechanism with:
    - Position (P): Location in semantic spacetime
    - Mass (M): Learnable importance weight
    - Force: Gravitational attraction between tokens
    
    Standard Attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    
    Gravitational Attention:
        Force(i,j) = G * (M_i * M_j) / (Distance(P_i, P_j)^2 + ε)
        Attention(P, M, V) = softmax(Force) V
    """
    
    def __init__(
        self,
        dim_model: int,
        dim_position: int,
        num_heads: int = 1,
        gravitational_constant: float = 1.0,
        event_horizon: float = 1e-6,
        max_force: Optional[float] = None,
        curvature: float = 0.0,
        learnable_G: bool = True,
        learnable_masses: bool = True
    ):
        """
        Initialize Gravitational Attention Layer.
        
        Args:
            dim_model: Model dimensionality (like d_model in Transformers)
            dim_position: Dimensionality of the semantic position space
            num_heads: Number of attention heads (multi-head attention)
            gravitational_constant: Initial value of G (learnable)
            event_horizon: ε to prevent singularities (division by zero)
            max_force: Maximum force (Hawking radiation limit), None for unbounded
            curvature: Spacetime curvature parameter (0 = flat, >0 = curved)
            learnable_G: Whether G is a learnable parameter
            learnable_masses: Whether masses are learnable per-token
        """
        self.dim_model = dim_model
        self.dim_position = dim_position
        self.num_heads = num_heads
        self.event_horizon = event_horizon
        self.max_force = max_force
        self.curvature = curvature
        
        # Gravitational constant (learnable)
        self.G = gravitational_constant
        self.learnable_G = learnable_G
        self.learnable_masses = learnable_masses
        
        # Projection matrices (replacing Q, K, V projections)
        # Position projection: maps input to semantic spacetime coordinates
        self.W_position = self._initialize_weight((dim_model, dim_position))
        
        # Mass projection: maps input to mass (importance)
        self.W_mass = self._initialize_weight((dim_model, 1))
        
        # Value projection: what gets transported by gravity
        self.W_value = self._initialize_weight((dim_model, dim_model))
        
        # Output projection
        self.W_output = self._initialize_weight((dim_model, dim_model))
        
        # Metric tensor for curved spacetime (if curvature > 0)
        if self.curvature > 0:
            self.metric_tensor = self._initialize_metric_tensor()
        else:
            self.metric_tensor = np.eye(dim_position)
    
    def _initialize_weight(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weight matrix with Xavier/Glorot initialization."""
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else 1
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def _initialize_metric_tensor(self) -> np.ndarray:
        """Initialize metric tensor for curved spacetime."""
        # Start with identity (flat space)
        metric = np.eye(self.dim_position)
        
        # Add curvature perturbations
        if self.curvature > 0:
            for i in range(self.dim_position):
                for j in range(self.dim_position):
                    if i != j:
                        # Schwarzschild-like curvature
                        metric[i, j] = self.curvature * np.exp(
                            -abs(i - j) / self.dim_position
                        )
        
        return metric
    
    def _compute_positions(self, X: np.ndarray) -> np.ndarray:
        """
        Compute semantic positions from input tokens.
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, dim_model)
            
        Returns:
            Positions tensor of shape (batch_size, seq_len, dim_position)
        """
        # Project to position space
        # (batch, seq_len, dim_model) @ (dim_model, dim_position)
        # -> (batch, seq_len, dim_position)
        positions = X @ self.W_position
        return positions
    
    def _compute_masses(self, X: np.ndarray) -> np.ndarray:
        """
        Compute masses (importance weights) from input tokens.
        
        Args:
            X: Input tensor of shape (batch_size, seq_len, dim_model)
            
        Returns:
            Masses tensor of shape (batch_size, seq_len, 1)
        """
        # Project to mass (importance)
        masses = X @ self.W_mass
        
        # Apply softplus to ensure positive masses: softplus(x) = log(1 + e^x)
        # This is crucial: mass must be positive!
        masses = np.log(1.0 + np.exp(masses))
        
        # Add minimum mass to prevent zero (numerical stability)
        masses = masses + 0.01
        
        return masses
    
    def _compute_geodesic_distance(
        self, 
        pos_i: np.ndarray, 
        pos_j: np.ndarray
    ) -> np.ndarray:
        """
        Compute geodesic distance on curved spacetime manifold.
        
        In flat space: d(i,j) = ||pos_i - pos_j||^2
        In curved space: d(i,j) = (pos_i - pos_j)^T M (pos_i - pos_j)
        
        Args:
            pos_i: Position of token i, shape (..., dim_position)
            pos_j: Position of token j, shape (..., dim_position)
            
        Returns:
            Squared geodesic distance, shape (...)
        """
        diff = pos_i - pos_j  # (..., dim_position)
        
        if self.curvature > 0:
            # Distance in curved space using metric tensor
            # d^2 = diff^T @ M @ diff
            temp = diff @ self.metric_tensor  # (..., dim_position)
            distance_squared = np.sum(temp * diff, axis=-1)  # (...)
        else:
            # Flat Euclidean distance
            distance_squared = np.sum(diff ** 2, axis=-1)  # (...)
        
        return distance_squared
    
    def _compute_gravitational_force(
        self,
        positions: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Compute gravitational force matrix between all token pairs.
        
        F(i,j) = G * (M_i * M_j) / (Distance(i,j)^2 + ε)
        
        Args:
            positions: Position tensor (batch_size, seq_len, dim_position)
            masses: Mass tensor (batch_size, seq_len, 1)
            
        Returns:
            Force matrix (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = positions.shape
        
        # Expand dimensions for broadcasting
        # positions_i: (batch, seq_len, 1, dim_position)
        # positions_j: (batch, 1, seq_len, dim_position)
        positions_i = positions[:, :, np.newaxis, :]  
        positions_j = positions[:, np.newaxis, :, :]
        
        # Compute pairwise distances
        # distance_matrix: (batch, seq_len, seq_len)
        distance_squared = self._compute_geodesic_distance(positions_i, positions_j)
        
        # Add event horizon to prevent singularities
        distance_squared = distance_squared + self.event_horizon
        
        # Compute mass products
        # masses: (batch, seq_len, 1)
        # mass_products: (batch, seq_len, seq_len)
        masses_i = masses[:, :, np.newaxis, :]  # (batch, seq_len, 1, 1)
        masses_j = masses[:, np.newaxis, :, :]  # (batch, 1, seq_len, 1)
        mass_products = masses_i * masses_j  # (batch, seq_len, seq_len, 1)
        mass_products = mass_products.squeeze(-1)  # (batch, seq_len, seq_len)
        
        # Gravitational force: F = G * (M1 * M2) / (d^2 + ε)
        force = self.G * mass_products / distance_squared
        
        # Apply Hawking Radiation limit (prevent black hole formation)
        if self.max_force is not None:
            force = np.minimum(force, self.max_force)
        
        return force
    
    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through Gravitational Attention layer.
        
        Args:
            X: Input tensor (batch_size, seq_len, dim_model)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
                  True = masked (no attention), False = attend
            
        Returns:
            Output tensor (batch_size, seq_len, dim_model)
        """
        batch_size, seq_len, _ = X.shape
        
        # 1. Compute semantic positions (where tokens exist in spacetime)
        positions = self._compute_positions(X)  # (batch, seq_len, dim_position)
        
        # 2. Compute masses (importance of each token)
        masses = self._compute_masses(X)  # (batch, seq_len, 1)
        
        # 3. Compute values (information to be transported by gravity)
        values = X @ self.W_value  # (batch, seq_len, dim_model)
        
        # 4. Compute gravitational force matrix
        force_matrix = self._compute_gravitational_force(
            positions, masses
        )  # (batch, seq_len, seq_len)
        
        # 5. Apply mask if provided (e.g., causal masking for autoregressive)
        if mask is not None:
            # Set masked positions to very large negative value
            force_matrix = np.where(mask, -1e9, force_matrix)
        
        # 6. Normalize forces (softmax) to get attention weights
        # This is the "probability" of information flow
        attention_weights = self._softmax(force_matrix, axis=-1)
        # (batch, seq_len, seq_len)
        
        # 7. Apply attention to values (gravitational transport of information)
        # output = attention_weights @ values
        output = attention_weights @ values  # (batch, seq_len, dim_model)
        
        # 8. Final output projection
        output = output @ self.W_output  # (batch, seq_len, dim_model)
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_attention_diagnostics(
        self,
        X: np.ndarray
    ) -> dict:
        """
        Get diagnostic information about attention mechanism.
        
        Args:
            X: Input tensor (batch_size, seq_len, dim_model)
            
        Returns:
            Dictionary with diagnostic information
        """
        positions = self._compute_positions(X)
        masses = self._compute_masses(X)
        force_matrix = self._compute_gravitational_force(positions, masses)
        
        return {
            'positions': positions,
            'masses': masses,
            'force_matrix': force_matrix,
            'gravitational_constant': self.G,
            'mean_mass': np.mean(masses),
            'max_mass': np.max(masses),
            'min_mass': np.min(masses),
            'mean_force': np.mean(force_matrix),
            'max_force': np.max(force_matrix),
            'spacetime_curvature': self.curvature
        }


class MultiHeadGravitationalAttention:
    """
    Multi-head version of Gravitational Attention.
    
    Like standard multi-head attention, this splits the attention into
    multiple "universes" (heads), each with its own gravitational laws.
    """
    
    def __init__(
        self,
        dim_model: int,
        dim_position: int,
        num_heads: int = 8,
        gravitational_constant: float = 1.0,
        event_horizon: float = 1e-6,
        max_force: Optional[float] = None,
        curvature: float = 0.0,
        different_G_per_head: bool = True
    ):
        """
        Initialize Multi-head Gravitational Attention.
        
        Args:
            dim_model: Model dimensionality
            dim_position: Position space dimensionality per head
            num_heads: Number of attention heads (parallel universes)
            gravitational_constant: Initial G value
            event_horizon: Singularity prevention parameter
            max_force: Hawking radiation limit
            curvature: Spacetime curvature
            different_G_per_head: Whether each head has its own G
        """
        assert dim_model % num_heads == 0, \
            f"dim_model ({dim_model}) must be divisible by num_heads ({num_heads})"
        
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_per_head = dim_model // num_heads
        
        # Create attention head for each "universe"
        self.heads = []
        for i in range(num_heads):
            # Each head can have different gravitational laws
            head_G = gravitational_constant
            if different_G_per_head:
                # Vary G across heads (different physics in each universe)
                head_G = gravitational_constant * (0.5 + i / num_heads)
            
            head = GravitationalAttentionLayer(
                dim_model=self.dim_per_head,
                dim_position=dim_position,
                num_heads=1,
                gravitational_constant=head_G,
                event_horizon=event_horizon,
                max_force=max_force,
                curvature=curvature
            )
            self.heads.append(head)
        
        # Final output projection to combine all heads
        self.W_output = self._initialize_weight((dim_model, dim_model))
    
    def _initialize_weight(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weight matrix."""
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through multi-head gravitational attention.
        
        Args:
            X: Input tensor (batch_size, seq_len, dim_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, dim_model)
        """
        batch_size, seq_len, _ = X.shape
        
        # Split input across heads
        # (batch, seq_len, dim_model) -> (batch, seq_len, num_heads, dim_per_head)
        X_split = X.reshape(batch_size, seq_len, self.num_heads, self.dim_per_head)
        
        # Process each head independently (parallel universes)
        head_outputs = []
        for i, head in enumerate(self.heads):
            # Extract this head's input
            X_head = X_split[:, :, i, :]  # (batch, seq_len, dim_per_head)
            
            # Apply gravitational attention
            output_head = head.forward(X_head, mask)
            head_outputs.append(output_head)
        
        # Concatenate all head outputs
        # List of (batch, seq_len, dim_per_head) -> (batch, seq_len, dim_model)
        output = np.concatenate(head_outputs, axis=-1)
        
        # Final projection
        output = output @ self.W_output
        
        return output
    
    def get_attention_diagnostics(self, X: np.ndarray) -> dict:
        """Get diagnostics from all attention heads."""
        batch_size, seq_len, _ = X.shape
        X_split = X.reshape(batch_size, seq_len, self.num_heads, self.dim_per_head)
        
        diagnostics = {}
        for i, head in enumerate(self.heads):
            X_head = X_split[:, :, i, :]
            head_diag = head.get_attention_diagnostics(X_head)
            diagnostics[f'head_{i}'] = head_diag
        
        return diagnostics


def demonstrate_gravitational_attention():
    """Demonstrate Gravitational Attention mechanism."""
    print("=" * 80)
    print("GRAVITATIONAL ATTENTION: Physics-Based Transformer Attention")
    print("=" * 80)
    print()
    
    # Setup
    batch_size = 2
    seq_len = 8
    dim_model = 64
    dim_position = 32
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {dim_model}")
    print(f"  Position dimension: {dim_position}")
    print()
    
    # Create sample input (e.g., token embeddings)
    X = np.random.randn(batch_size, seq_len, dim_model) * 0.5
    
    print("-" * 80)
    print("Single-Head Gravitational Attention")
    print("-" * 80)
    
    # Initialize gravitational attention
    grav_attn = GravitationalAttentionLayer(
        dim_model=dim_model,
        dim_position=dim_position,
        gravitational_constant=1.0,
        event_horizon=1e-6,
        max_force=100.0,  # Hawking radiation limit
        curvature=0.1
    )
    
    # Forward pass
    output = grav_attn.forward(X)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {np.linalg.norm(output):.4f}")
    
    # Get diagnostics
    diagnostics = grav_attn.get_attention_diagnostics(X)
    print(f"\nGravitational Diagnostics:")
    print(f"  G (gravitational constant): {diagnostics['gravitational_constant']:.4f}")
    print(f"  Mean mass: {diagnostics['mean_mass']:.4f}")
    print(f"  Max mass: {diagnostics['max_mass']:.4f}")
    print(f"  Min mass: {diagnostics['min_mass']:.4f}")
    print(f"  Mean force: {diagnostics['mean_force']:.4f}")
    print(f"  Max force: {diagnostics['max_force']:.4f}")
    print(f"  Spacetime curvature: {diagnostics['spacetime_curvature']:.4f}")
    
    # Show force matrix for first batch
    force_matrix = diagnostics['force_matrix'][0]
    print(f"\nForce Matrix (first sequence):")
    print(f"  Shape: {force_matrix.shape}")
    print(f"  Sample (first 3x3):")
    print(force_matrix[:3, :3])
    
    print("\n" + "-" * 80)
    print("Multi-Head Gravitational Attention")
    print("-" * 80)
    
    # Multi-head version
    multi_grav_attn = MultiHeadGravitationalAttention(
        dim_model=dim_model,
        dim_position=16,  # Smaller per head
        num_heads=4,
        gravitational_constant=1.0,
        max_force=100.0,
        curvature=0.15,
        different_G_per_head=True
    )
    
    output_multi = multi_grav_attn.forward(X)
    
    print(f"\nInput shape: {X.shape}")
    print(f"Output shape: {output_multi.shape}")
    print(f"Output norm: {np.linalg.norm(output_multi):.4f}")
    
    # Diagnostics for each head
    multi_diagnostics = multi_grav_attn.get_attention_diagnostics(X)
    print(f"\nMulti-Head Diagnostics:")
    for head_name, head_diag in multi_diagnostics.items():
        print(f"  {head_name}:")
        print(f"    G: {head_diag['gravitational_constant']:.4f}")
        print(f"    Mean force: {head_diag['mean_force']:.4f}")
        print(f"    Max force: {head_diag['max_force']:.4f}")
    
    print("\n" + "-" * 80)
    print("Comparison: Massive vs Light Tokens")
    print("-" * 80)
    
    # Create input where one token has much higher activation (will have high mass)
    X_test = np.random.randn(1, 5, dim_model) * 0.3
    X_test[0, 2, :] *= 5.0  # Token 2 is "massive"
    
    output_test = grav_attn.forward(X_test)
    diag_test = grav_attn.get_attention_diagnostics(X_test)
    
    print(f"\nToken masses:")
    for i, mass in enumerate(diag_test['masses'][0]):
        print(f"  Token {i}: mass = {mass[0]:.4f}")
    
    print(f"\nGravitational forces FROM token 2 (massive) TO others:")
    force_from_massive = diag_test['force_matrix'][0, 2, :]
    for i, force in enumerate(force_from_massive):
        print(f"  Token 2 -> Token {i}: force = {force:.4f}")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_gravitational_attention()
