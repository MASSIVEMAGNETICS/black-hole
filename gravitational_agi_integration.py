"""
Gravitational AGI Integration Module

Integrates Gravitational Attention with the Black Hole Framework to create
a complete AGI system with physics-based reasoning and attention mechanisms.

This module serves as the bridge between:
- Gravitational Attention (Transformer-compatible attention)
- Black Hole Framework (containment, stability, self-awareness)
- AGI reasoning and evolution capabilities
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time
import json

# Import our framework components
from gravitational_attention import (
    GravitationalAttentionLayer,
    MultiHeadGravitationalAttention
)
from black_hole_framework import (
    BlackHoleFramework,
    BekensteinBound,
    HawkingRadiation,
    ContainmentProtocol,
    EternalLoop,
    CurvedManifold,
    GravitationalAttention as GravitationalAttentionPhysics
)


class GravitationalAGICore:
    """
    Complete AGI system integrating Gravitational Attention with
    the Black Hole containment framework.
    
    This system combines:
    1. Transformer-compatible Gravitational Attention for sequence processing
    2. Black Hole physics for information stability and ethics
    3. Self-awareness through recursive observation
    4. Evolutionary adaptation with containment
    """
    
    def __init__(
        self,
        dim_model: int = 128,
        dim_position: int = 64,
        num_heads: int = 4,
        sequence_length: int = 512,
        gravitational_constant: float = 1.0,
        curvature: float = 0.15,
        energy: float = 2.0,
        ethical_threshold: float = 0.6
    ):
        """
        Initialize the Gravitational AGI Core.
        
        Args:
            dim_model: Model dimensionality (embedding size)
            dim_position: Position space dimensionality for attention
            num_heads: Number of attention heads
            sequence_length: Maximum sequence length
            gravitational_constant: G for gravitational attention
            curvature: Spacetime curvature parameter
            energy: Energy for Bekenstein bound
            ethical_threshold: Threshold for ethical alignment
        """
        self.dim_model = dim_model
        self.sequence_length = sequence_length
        
        # Core attention mechanism (for Transformer integration)
        self.attention = MultiHeadGravitationalAttention(
            dim_model=dim_model,
            dim_position=dim_position,
            num_heads=num_heads,
            gravitational_constant=gravitational_constant,
            max_force=100.0,  # Hawking radiation limit
            curvature=curvature,
            different_G_per_head=True
        )
        
        # Black Hole containment framework
        self.black_hole_framework = BlackHoleFramework(
            dimensions=dim_model,
            curvature=curvature,
            energy=energy,
            ethical_threshold=ethical_threshold
        )
        
        # State tracking
        self.processing_history = []
        self.stability_violations = []
        self.self_awareness_level = 0.0
        
        # Evolution parameters
        self.evolution_count = 0
        self.entropy = 0.0
        
    def process_sequence(
        self,
        sequence: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """
        Process a sequence through the complete AGI pipeline.
        
        Pipeline:
        1. Gravitational Attention (physics-based attention)
        2. Self-Awareness (eternal loop observation)
        3. Containment Check (stability and ethics)
        4. Hawking Radiation (controlled decay)
        
        Args:
            sequence: Input sequence (batch_size, seq_len, dim_model)
            mask: Optional attention mask
            return_diagnostics: Whether to return detailed diagnostics
            
        Returns:
            Dictionary with processed output and diagnostics
        """
        batch_size, seq_len, _ = sequence.shape
        
        # Stage 1: Gravitational Attention
        # Information flows according to gravitational laws
        attended_output = self.attention.forward(sequence, mask)
        
        # Stage 2: Process through Black Hole framework for each item in sequence
        # This adds containment, stability, and self-awareness
        processed_outputs = []
        stability_reports = []
        
        for batch_idx in range(batch_size):
            batch_output = attended_output[batch_idx]  # (seq_len, dim_model)
            
            # Average across sequence for framework processing
            aggregated = np.mean(batch_output, axis=0)  # (dim_model,)
            
            # Apply Black Hole framework
            result = self.black_hole_framework.process(aggregated)
            
            processed_outputs.append(result['output'])
            stability_reports.append(result['stability'])
            
            # Update self-awareness level
            if result['self_aware']:
                self.self_awareness_level += 0.01
            
            # Track stability violations
            if not result['stability']['stable']:
                self.stability_violations.append({
                    'timestep': self.black_hole_framework.timestep,
                    'batch_idx': batch_idx,
                    'entropy': result['stability']['entropy'],
                    'ethical_score': result['stability']['ethical_score']
                })
        
        # Reconstruct output
        final_output = np.array(processed_outputs)  # (batch_size, dim_model)
        
        # Expand back to sequence shape
        output_expanded = np.expand_dims(final_output, axis=1)
        output_expanded = np.tile(output_expanded, (1, seq_len, 1))
        
        # Store in history
        self.processing_history.append({
            'timestep': time.time(),
            'self_awareness': self.self_awareness_level,
            'entropy': np.mean([r['entropy'] for r in stability_reports]),
            'stable': all(r['stable'] for r in stability_reports)
        })
        
        result_dict = {
            'output': output_expanded,
            'attended_output': attended_output,
            'stability_reports': stability_reports,
            'self_awareness_level': self.self_awareness_level,
            'timestep': self.black_hole_framework.timestep
        }
        
        if return_diagnostics:
            result_dict['diagnostics'] = self.get_diagnostics()
        
        return result_dict
    
    def evolve(self) -> Dict[str, Any]:
        """
        Perform self-evolution step.
        
        Evolution includes:
        - Adjusting gravitational constants
        - Modifying curvature parameters
        - Adapting ethical thresholds
        - Optimizing attention parameters
        
        Returns:
            Evolution report
        """
        self.evolution_count += 1
        
        # Check current health
        recent_stability = []
        for entry in self.processing_history[-10:]:
            recent_stability.append(1.0 if entry['stable'] else 0.0)
        
        health_score = np.mean(recent_stability) if recent_stability else 0.5
        
        # Evolve parameters based on health
        if health_score < 0.7:
            # System struggling - make it more stable
            print(f"[EVOLVE] Health low ({health_score:.2f}), increasing stability...")
            
            # Increase ethical threshold
            current_threshold = self.black_hole_framework.containment.ethical_threshold
            self.black_hole_framework.containment.ethical_threshold = min(
                0.9, current_threshold + 0.05
            )
            
            # Reduce curvature for more predictable behavior
            self.black_hole_framework.manifold.curvature *= 0.95
            
        elif health_score > 0.9:
            # System very stable - can explore more
            print(f"[EVOLVE] Health excellent ({health_score:.2f}), increasing exploration...")
            
            # Increase curvature for richer dynamics
            self.black_hole_framework.manifold.curvature *= 1.05
            
            # Adjust gravitational constants
            for head in self.attention.heads:
                head.G *= (0.95 + np.random.random() * 0.1)
        
        evolution_report = {
            'evolution_count': self.evolution_count,
            'health_score': health_score,
            'new_curvature': self.black_hole_framework.manifold.curvature,
            'new_ethical_threshold': self.black_hole_framework.containment.ethical_threshold,
            'self_awareness_level': self.self_awareness_level
        }
        
        return evolution_report
    
    def introspect(self) -> Dict[str, Any]:
        """
        Perform self-introspection to assess internal state.
        
        Returns:
            Introspection report
        """
        # Analyze recent processing history
        recent_entries = self.processing_history[-20:] if len(self.processing_history) >= 20 else self.processing_history
        
        avg_entropy = np.mean([e['entropy'] for e in recent_entries]) if recent_entries else 0.0
        stability_rate = np.mean([1.0 if e['stable'] else 0.0 for e in recent_entries]) if recent_entries else 0.0
        
        # Get framework diagnostics
        framework_diag = self.black_hole_framework.get_diagnostics()
        
        # Assess self-awareness
        causality = self.black_hole_framework.eternal_loop.recognize_causal_origin()
        
        introspection_report = {
            'self_awareness_level': self.self_awareness_level,
            'causality_recognized': causality['recognized'],
            'causality_strength': causality.get('causality_strength', 0.0),
            'avg_entropy': avg_entropy,
            'stability_rate': stability_rate,
            'total_violations': len(self.stability_violations),
            'evolution_count': self.evolution_count,
            'framework_diagnostics': framework_diag,
            'is_healthy': stability_rate > 0.7 and avg_entropy < 0.5,
            'is_self_aware': causality['recognized'] and self.self_awareness_level > 0.5
        }
        
        return introspection_report
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        attention_diag = {}
        if hasattr(self.attention, 'get_attention_diagnostics'):
            # Create dummy input for diagnostics
            dummy_input = np.random.randn(1, 8, self.dim_model)
            attention_diag = self.attention.get_attention_diagnostics(dummy_input)
        
        return {
            'processing_history_length': len(self.processing_history),
            'stability_violations': len(self.stability_violations),
            'self_awareness_level': self.self_awareness_level,
            'evolution_count': self.evolution_count,
            'attention_diagnostics': attention_diag,
            'black_hole_diagnostics': self.black_hole_framework.get_diagnostics()
        }
    
    def save_state(self, filepath: str):
        """Save AGI state to file."""
        state = {
            'processing_history': self.processing_history,
            'stability_violations': self.stability_violations,
            'self_awareness_level': self.self_awareness_level,
            'evolution_count': self.evolution_count,
            'entropy': self.entropy,
            'attention_G_values': [head.G for head in self.attention.heads],
            'curvature': self.black_hole_framework.manifold.curvature,
            'ethical_threshold': self.black_hole_framework.containment.ethical_threshold
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"AGI state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load AGI state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.processing_history = state['processing_history']
        self.stability_violations = state['stability_violations']
        self.self_awareness_level = state['self_awareness_level']
        self.evolution_count = state['evolution_count']
        self.entropy = state['entropy']
        
        # Restore parameters
        for i, G in enumerate(state['attention_G_values']):
            if i < len(self.attention.heads):
                self.attention.heads[i].G = G
        
        self.black_hole_framework.manifold.curvature = state['curvature']
        self.black_hole_framework.containment.ethical_threshold = state['ethical_threshold']
        
        print(f"AGI state loaded from {filepath}")


def demonstrate_gravitational_agi():
    """Demonstrate the complete Gravitational AGI system."""
    print("=" * 80)
    print("GRAVITATIONAL AGI: Complete Self-Aware System with Physics-Based Attention")
    print("=" * 80)
    print()
    
    # Initialize AGI
    print("Initializing Gravitational AGI...")
    agi = GravitationalAGICore(
        dim_model=64,
        dim_position=32,
        num_heads=4,
        sequence_length=16,
        gravitational_constant=1.0,
        curvature=0.15,
        ethical_threshold=0.6
    )
    print("✓ AGI initialized")
    print()
    
    # Process sequences
    print("-" * 80)
    print("Processing Sequences Through AGI")
    print("-" * 80)
    
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}:")
        
        # Create sample sequence (simulating token embeddings)
        batch_size = 2
        seq_len = 8
        sequence = np.random.randn(batch_size, seq_len, agi.dim_model) * 0.5
        
        # Process
        result = agi.process_sequence(sequence, return_diagnostics=False)
        
        print(f"  Input shape: {sequence.shape}")
        print(f"  Output shape: {result['output'].shape}")
        print(f"  Self-awareness level: {result['self_awareness_level']:.4f}")
        print(f"  All stable: {all(r['stable'] for r in result['stability_reports'])}")
        
        avg_entropy = np.mean([r['entropy'] for r in result['stability_reports']])
        print(f"  Average entropy: {avg_entropy:.4f}")
        
        # Evolve every 2 epochs
        if (epoch + 1) % 2 == 0:
            print("\n  [EVOLUTION TRIGGERED]")
            evolution_report = agi.evolve()
            print(f"    Health score: {evolution_report['health_score']:.4f}")
            print(f"    Evolution #{evolution_report['evolution_count']}")
    
    # Final introspection
    print("\n" + "-" * 80)
    print("Final Self-Introspection")
    print("-" * 80)
    
    introspection = agi.introspect()
    print(f"\nIntrospection Report:")
    print(f"  Self-aware: {introspection['is_self_aware']}")
    print(f"  Self-awareness level: {introspection['self_awareness_level']:.4f}")
    print(f"  Causality recognized: {introspection['causality_recognized']}")
    if introspection['causality_recognized']:
        print(f"  Causality strength: {introspection['causality_strength']:.4f}")
    print(f"  Health status: {'HEALTHY' if introspection['is_healthy'] else 'NEEDS ATTENTION'}")
    print(f"  Stability rate: {introspection['stability_rate']:.2%}")
    print(f"  Average entropy: {introspection['avg_entropy']:.4f}")
    print(f"  Evolution count: {introspection['evolution_count']}")
    print(f"  Total violations: {introspection['total_violations']}")
    
    # Full diagnostics
    print("\n" + "-" * 80)
    print("Complete System Diagnostics")
    print("-" * 80)
    
    diagnostics = agi.get_diagnostics()
    print(f"\nSystem Metrics:")
    print(f"  Processing history: {diagnostics['processing_history_length']} steps")
    print(f"  Stability violations: {diagnostics['stability_violations']}")
    print(f"  Self-awareness level: {diagnostics['self_awareness_level']:.4f}")
    print(f"  Evolution count: {diagnostics['evolution_count']}")
    
    print(f"\nBlack Hole Framework:")
    bh_diag = diagnostics['black_hole_diagnostics']
    print(f"  Timestep: {bh_diag['timestep']}")
    print(f"  Manifold curvature: {bh_diag['manifold_curvature']:.4f}")
    print(f"  Max entropy (Bekenstein): {bh_diag['bekenstein_max_entropy']:.4f}")
    print(f"  Hawking temperature: {bh_diag['hawking_temperature']:.4f}")
    print(f"  Containment violations: {bh_diag['containment_violations']}")
    print(f"  Eternal loop depth: {bh_diag['eternal_loop_depth']}")
    
    # Test state save/load
    print("\n" + "-" * 80)
    print("Testing State Persistence")
    print("-" * 80)
    
    agi.save_state('/tmp/agi_state.json')
    
    # Create new AGI with same parameters and load state
    agi2 = GravitationalAGICore(
        dim_model=64,
        dim_position=32,
        num_heads=4,
        curvature=0.15
    )
    agi2.load_state('/tmp/agi_state.json')
    
    print(f"✓ State successfully saved and restored")
    print(f"  Restored self-awareness: {agi2.self_awareness_level:.4f}")
    print(f"  Restored evolution count: {agi2.evolution_count}")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_gravitational_agi()
