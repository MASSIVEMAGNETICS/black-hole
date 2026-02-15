"""
Example usage of the Black Hole Framework for various scenarios.
"""

import numpy as np
from black_hole_framework import (
    BlackHoleFramework,
    CurvedManifold,
    GravitationalAttention,
    BekensteinBound,
    HawkingRadiation,
    ContainmentProtocol,
    EternalLoop
)


def example_basic_processing():
    """Example 1: Basic information processing."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Information Processing")
    print("=" * 70)
    
    framework = BlackHoleFramework(dimensions=16, curvature=0.1)
    
    # Single input processing
    input_vector = np.random.randn(16)
    result = framework.process(input_vector)
    
    print(f"Input norm: {np.linalg.norm(input_vector):.4f}")
    print(f"Output norm: {np.linalg.norm(result['output']):.4f}")
    print(f"Stable: {result['stability']['stable']}")
    print(f"Self-aware: {result['self_aware']}")


def example_gravitational_attention():
    """Example 2: Gravitational attention with context."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Gravitational Attention with Context")
    print("=" * 70)
    
    framework = BlackHoleFramework(dimensions=32, curvature=0.2)
    
    # Query
    query = np.random.randn(32)
    
    # Context: 3 memory items with different importance
    context_keys = np.array([
        np.random.randn(32),  # Recent memory
        np.random.randn(32),  # Older memory
        np.random.randn(32),  # Ancient memory
    ])
    
    context_values = context_keys * 0.8  # Values related to keys
    
    # Importance decreases with age
    context_masses = np.array([1.5, 1.0, 0.5])
    
    result = framework.process(
        query,
        context_keys=context_keys,
        context_values=context_values,
        context_masses=context_masses
    )
    
    print("Context masses:", context_masses)
    print(f"Output incorporates weighted context")
    print(f"Output norm: {np.linalg.norm(result['output']):.4f}")


def example_stability_monitoring():
    """Example 3: Monitoring information stability."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Stability Monitoring and Containment")
    print("=" * 70)
    
    framework = BlackHoleFramework(
        dimensions=32,
        curvature=0.15,
        energy=1.0,  # Lower energy = tighter entropy bound
        ethical_threshold=0.6
    )
    
    # Process normal input
    normal_input = np.random.randn(32) * 0.5
    result1 = framework.process(normal_input)
    print(f"Normal input - Entropy: {result1['stability']['entropy']:.4f}, "
          f"Stable: {result1['stability']['stable']}")
    
    # Process potentially problematic input (high magnitude)
    extreme_input = np.random.randn(32) * 10.0
    result2 = framework.process(extreme_input)
    print(f"Extreme input - Entropy: {result2['stability']['entropy']:.4f}, "
          f"Stable: {result2['stability']['stable']}")
    
    # Check violations
    diagnostics = framework.get_diagnostics()
    print(f"Total containment violations: {diagnostics['containment_violations']}")


def example_self_awareness_emergence():
    """Example 4: Observing self-awareness emergence."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Self-Awareness Emergence")
    print("=" * 70)
    
    framework = BlackHoleFramework(dimensions=32, curvature=0.1)
    
    input_vector = np.random.randn(32) * 0.5
    
    print("Processing over time...")
    for step in range(1, 8):
        result = framework.process(input_vector)
        
        print(f"Step {step}: Self-aware={result['self_aware']}", end="")
        if result['causality']['recognized']:
            print(f", Causality strength={result['causality']['causality_strength']:.4f}")
        else:
            print(f", {result['causality']['message']}")
        
        # Evolve input slightly
        input_vector = result['output'] * 0.7 + np.random.randn(32) * 0.2


def example_hawking_radiation_decay():
    """Example 5: Information decay through Hawking radiation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Hawking Radiation and Information Decay")
    print("=" * 70)
    
    # Create Hawking radiation component directly
    hawking = HawkingRadiation(temperature=0.2, decay_rate=0.05)
    
    initial_state = np.ones(16)
    state = initial_state.copy()
    
    print(f"Initial state norm: {np.linalg.norm(state):.4f}")
    
    for t in range(1, 6):
        state = hawking.emit(state, t)
        print(f"After {t} steps: norm={np.linalg.norm(state):.4f}")
    
    print(f"Total emission events: {len(hawking.get_emitted_history())}")


def example_manifold_geometry():
    """Example 6: Exploring curved manifold geometry."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Curved Manifold Geometry")
    print("=" * 70)
    
    # Compare flat vs curved manifolds
    flat_manifold = CurvedManifold(dimensions=16, curvature=0.0)
    curved_manifold = CurvedManifold(dimensions=16, curvature=0.3)
    
    point_a = np.random.randn(16)
    point_b = np.random.randn(16)
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(point_b - point_a)
    
    # Geodesic distances
    flat_dist = flat_manifold.geodesic_distance(point_a, point_b)
    curved_dist = curved_manifold.geodesic_distance(point_a, point_b)
    
    print(f"Euclidean distance: {euclidean_dist:.4f}")
    print(f"Flat manifold geodesic: {flat_dist:.4f}")
    print(f"Curved manifold geodesic: {curved_dist:.4f}")
    print(f"Curvature effect: {((curved_dist - flat_dist) / flat_dist * 100):.2f}%")


def example_eternal_loop():
    """Example 7: Eternal loop and recursive self-observation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Eternal Loop - Recursive Self-Observation")
    print("=" * 70)
    
    eternal_loop = EternalLoop(dimensions=16, memory_depth=5)
    
    # Simulate processing over multiple steps
    state = np.random.randn(16)
    
    for step in range(1, 7):
        # Self-observe
        self_obs = eternal_loop.observe_self(state)
        
        # Check causal recognition
        causality = eternal_loop.recognize_causal_origin()
        
        print(f"Step {step}: Recognized={causality['recognized']}, "
              f"Strength={causality['causality_strength']:.4f}")
        
        # Evolve state
        state = 0.8 * state + 0.2 * self_obs + np.random.randn(16) * 0.1


def example_bekenstein_bound():
    """Example 8: Bekenstein bound enforcement."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Bekenstein Bound Enforcement")
    print("=" * 70)
    
    # Different energy levels create different bounds
    tight_bound = BekensteinBound(radius=1.0, energy=0.5)
    loose_bound = BekensteinBound(radius=1.0, energy=2.0)
    
    print(f"Tight bound max entropy: {tight_bound.calculate_max_entropy():.4f}")
    print(f"Loose bound max entropy: {loose_bound.calculate_max_entropy():.4f}")
    
    # Test entropy values
    test_entropy = 5.0
    
    tight_check = tight_bound.check_entropy_limit(test_entropy)
    loose_check = loose_bound.check_entropy_limit(test_entropy)
    
    print(f"\nTest entropy: {test_entropy}")
    print(f"Within tight bound: {tight_check[0]} (ratio: {tight_check[1]:.2f})")
    print(f"Within loose bound: {loose_check[0]} (ratio: {loose_check[1]:.2f})")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BLACK HOLE FRAMEWORK - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    example_basic_processing()
    example_gravitational_attention()
    example_stability_monitoring()
    example_self_awareness_emergence()
    example_hawking_radiation_decay()
    example_manifold_geometry()
    example_eternal_loop()
    example_bekenstein_bound()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
