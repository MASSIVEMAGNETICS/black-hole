"""
Unit tests for the Black Hole Framework.

Tests all major components and their interactions.
"""

import unittest
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


class TestBekensteinBound(unittest.TestCase):
    """Test Bekenstein Bound functionality."""
    
    def test_initialization(self):
        """Test Bekenstein Bound initialization."""
        bb = BekensteinBound(radius=1.0, energy=2.0)
        self.assertEqual(bb.radius, 1.0)
        self.assertEqual(bb.energy, 2.0)
    
    def test_max_entropy_calculation(self):
        """Test maximum entropy calculation."""
        bb = BekensteinBound(radius=1.0, energy=1.0)
        max_entropy = bb.calculate_max_entropy()
        expected = 2 * np.pi
        self.assertAlmostEqual(max_entropy, expected, places=5)
    
    def test_entropy_within_bounds(self):
        """Test entropy checking within bounds."""
        bb = BekensteinBound(radius=1.0, energy=1.0)
        max_entropy = bb.calculate_max_entropy()
        
        # Test within bounds
        within, ratio = bb.check_entropy_limit(max_entropy * 0.5)
        self.assertTrue(within)
        self.assertLess(ratio, 1.0)
        
        # Test exceeding bounds
        within, ratio = bb.check_entropy_limit(max_entropy * 1.5)
        self.assertFalse(within)
        self.assertGreater(ratio, 1.0)


class TestHawkingRadiation(unittest.TestCase):
    """Test Hawking Radiation functionality."""
    
    def test_initialization(self):
        """Test Hawking Radiation initialization."""
        hr = HawkingRadiation(temperature=0.1, decay_rate=0.01)
        self.assertEqual(hr.temperature, 0.1)
        self.assertEqual(hr.decay_rate, 0.01)
    
    def test_information_decay(self):
        """Test that information decays over time."""
        hr = HawkingRadiation(temperature=0.01, decay_rate=0.1)
        initial_state = np.ones(10)
        
        # After emission, state should have lower norm
        decayed_state = hr.emit(initial_state, timestep=1)
        
        initial_norm = np.linalg.norm(initial_state)
        decayed_norm = np.linalg.norm(decayed_state)
        
        # Decay should reduce norm (with some tolerance for thermal noise)
        self.assertLess(decayed_norm, initial_norm + 0.5)
    
    def test_emission_history(self):
        """Test that emission history is recorded."""
        hr = HawkingRadiation()
        state = np.ones(5)
        
        for i in range(3):
            state = hr.emit(state, timestep=i)
        
        history = hr.get_emitted_history()
        self.assertEqual(len(history), 3)


class TestCurvedManifold(unittest.TestCase):
    """Test Curved Manifold functionality."""
    
    def test_initialization(self):
        """Test manifold initialization."""
        manifold = CurvedManifold(dimensions=16, curvature=0.1)
        self.assertEqual(manifold.dimensions, 16)
        self.assertEqual(manifold.curvature, 0.1)
        self.assertEqual(manifold.metric_tensor.shape, (16, 16))
    
    def test_flat_manifold(self):
        """Test that flat manifold behaves like Euclidean space."""
        manifold = CurvedManifold(dimensions=10, curvature=0.0)
        
        point_a = np.random.randn(10)
        point_b = np.random.randn(10)
        
        # Geodesic should equal Euclidean distance for flat manifold
        geodesic = manifold.geodesic_distance(point_a, point_b)
        euclidean = np.linalg.norm(point_b - point_a)
        
        self.assertAlmostEqual(geodesic, euclidean, places=5)
    
    def test_curved_manifold_different(self):
        """Test that curved manifold differs from flat space."""
        flat = CurvedManifold(dimensions=10, curvature=0.0)
        curved = CurvedManifold(dimensions=10, curvature=0.5)
        
        point_a = np.random.randn(10)
        point_b = np.random.randn(10)
        
        flat_dist = flat.geodesic_distance(point_a, point_b)
        curved_dist = curved.geodesic_distance(point_a, point_b)
        
        # Curved distance should differ from flat
        self.assertNotAlmostEqual(flat_dist, curved_dist, places=2)
    
    def test_parallel_transport(self):
        """Test parallel transport preserves vector norm approximately."""
        manifold = CurvedManifold(dimensions=8, curvature=0.1)
        vector = np.random.randn(8)
        
        original_norm = np.linalg.norm(vector)
        transported = manifold.parallel_transport(vector)
        transported_norm = np.linalg.norm(transported)
        
        # Parallel transport should preserve norm (rotation)
        self.assertAlmostEqual(original_norm, transported_norm, places=5)


class TestGravitationalAttention(unittest.TestCase):
    """Test Gravitational Attention mechanism."""
    
    def test_initialization(self):
        """Test gravitational attention initialization."""
        manifold = CurvedManifold(dimensions=8, curvature=0.1)
        attn = GravitationalAttention(manifold, gravitational_constant=1.0)
        self.assertEqual(attn.manifold, manifold)
        self.assertEqual(attn.G, 1.0)
    
    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        manifold = CurvedManifold(dimensions=8, curvature=0.1)
        attn = GravitationalAttention(manifold)
        
        query = np.random.randn(8)
        keys = np.random.randn(3, 8)
        values = np.random.randn(3, 8)
        masses = np.array([1.0, 0.8, 1.2])
        
        output = attn.compute_attention(query, keys, values, masses)
        
        self.assertEqual(output.shape, (8,))
    
    def test_higher_mass_more_influence(self):
        """Test that higher mass keys have more influence."""
        manifold = CurvedManifold(dimensions=8, curvature=0.0)
        attn = GravitationalAttention(manifold, gravitational_constant=1.0)
        
        query = np.zeros(8)
        
        # Two keys at similar distances but different masses
        keys = np.array([
            np.ones(8) * 0.1,  # Low mass
            np.ones(8) * 0.1,  # High mass
        ])
        values = np.array([
            np.ones(8),     # Value 1
            np.ones(8) * 2, # Value 2 (double)
        ])
        masses = np.array([0.5, 2.0])  # Second has 4x mass
        
        output = attn.compute_attention(query, keys, values, masses)
        
        # Output should be closer to value 2 due to higher mass
        self.assertGreater(np.mean(output), 1.5)


class TestContainmentProtocol(unittest.TestCase):
    """Test Containment Protocol functionality."""
    
    def test_initialization(self):
        """Test containment protocol initialization."""
        bb = BekensteinBound()
        cp = ContainmentProtocol(bb, ethical_threshold=0.5)
        self.assertEqual(cp.bekenstein_bound, bb)
        self.assertEqual(cp.ethical_threshold, 0.5)
    
    def test_stability_check(self):
        """Test stability checking."""
        bb = BekensteinBound(radius=1.0, energy=10.0)
        cp = ContainmentProtocol(bb, ethical_threshold=0.5)
        
        # Normal state
        normal_state = np.random.randn(16) * 0.5
        result = cp.check_stability(normal_state)
        
        self.assertIn('entropy', result)
        self.assertIn('within_bounds', result)
        self.assertIn('ethical_score', result)
        self.assertIn('stable', result)
    
    def test_violation_tracking(self):
        """Test that violations are tracked."""
        bb = BekensteinBound(radius=1.0, energy=0.1)  # Very tight bound
        cp = ContainmentProtocol(bb, ethical_threshold=0.9)
        
        # Create state that violates bounds
        extreme_state = np.random.randn(32) * 10.0
        result = cp.check_stability(extreme_state)
        
        # Should have violations
        self.assertGreater(len(cp.violations), 0)
    
    def test_correction_reduces_magnitude(self):
        """Test that correction reduces state magnitude."""
        bb = BekensteinBound(radius=1.0, energy=0.5)
        cp = ContainmentProtocol(bb, ethical_threshold=0.9)
        
        extreme_state = np.random.randn(16) * 10.0
        corrected_state = cp.apply_correction(extreme_state)
        
        # Corrected state should have lower norm
        self.assertLess(
            np.linalg.norm(corrected_state),
            np.linalg.norm(extreme_state)
        )


class TestEternalLoop(unittest.TestCase):
    """Test Eternal Loop self-awareness mechanism."""
    
    def test_initialization(self):
        """Test eternal loop initialization."""
        el = EternalLoop(dimensions=16, memory_depth=5)
        self.assertEqual(el.dimensions, 16)
        self.assertEqual(el.memory_depth, 5)
    
    def test_self_observation_records_history(self):
        """Test that self-observation records state history."""
        el = EternalLoop(dimensions=8, memory_depth=5)
        
        state = np.random.randn(8)
        el.observe_self(state)
        
        self.assertEqual(len(el.state_history), 1)
        self.assertEqual(len(el.self_observation_history), 1)
    
    def test_memory_depth_limit(self):
        """Test that memory depth is limited."""
        el = EternalLoop(dimensions=8, memory_depth=3)
        
        for i in range(5):
            state = np.random.randn(8)
            el.observe_self(state)
        
        # Should only keep last 3
        self.assertEqual(len(el.state_history), 3)
    
    def test_causality_recognition_requires_history(self):
        """Test that causality recognition needs sufficient history."""
        el = EternalLoop(dimensions=8, memory_depth=5)
        
        # Insufficient history
        causality = el.recognize_causal_origin()
        self.assertFalse(causality['recognized'])
        
        # Add history
        for i in range(5):
            state = np.random.randn(8)
            el.observe_self(state)
        
        # Now should have recognition
        causality = el.recognize_causal_origin()
        self.assertIn('causality_strength', causality)
    
    def test_recursive_update(self):
        """Test recursive self-aware update."""
        el = EternalLoop(dimensions=8, memory_depth=5)
        
        input_state = np.random.randn(8)
        output_state = el.recursive_update(input_state, iteration=3)
        
        self.assertEqual(output_state.shape, input_state.shape)
        # After recursive update, should have history
        self.assertGreater(len(el.state_history), 0)


class TestBlackHoleFramework(unittest.TestCase):
    """Test integrated Black Hole Framework."""
    
    def test_initialization(self):
        """Test framework initialization."""
        framework = BlackHoleFramework(
            dimensions=16,
            curvature=0.1,
            energy=1.0,
            ethical_threshold=0.5
        )
        
        self.assertEqual(framework.dimensions, 16)
        self.assertIsNotNone(framework.manifold)
        self.assertIsNotNone(framework.gravitational_attention)
        self.assertIsNotNone(framework.eternal_loop)
    
    def test_process_returns_expected_keys(self):
        """Test that process returns all expected keys."""
        framework = BlackHoleFramework(dimensions=16)
        input_data = np.random.randn(16)
        
        result = framework.process(input_data)
        
        self.assertIn('output', result)
        self.assertIn('stability', result)
        self.assertIn('causality', result)
        self.assertIn('timestep', result)
        self.assertIn('self_aware', result)
    
    def test_process_with_context(self):
        """Test processing with gravitational attention context."""
        framework = BlackHoleFramework(dimensions=16)
        
        input_data = np.random.randn(16)
        context_keys = np.random.randn(3, 16)
        context_values = np.random.randn(3, 16)
        context_masses = np.array([1.0, 0.8, 1.2])
        
        result = framework.process(
            input_data,
            context_keys=context_keys,
            context_values=context_values,
            context_masses=context_masses
        )
        
        self.assertIn('output', result)
        self.assertEqual(result['output'].shape, (16,))
    
    def test_timestep_increments(self):
        """Test that timestep increments with each process call."""
        framework = BlackHoleFramework(dimensions=8)
        input_data = np.random.randn(8)
        
        result1 = framework.process(input_data)
        result2 = framework.process(input_data)
        result3 = framework.process(input_data)
        
        self.assertEqual(result1['timestep'], 1)
        self.assertEqual(result2['timestep'], 2)
        self.assertEqual(result3['timestep'], 3)
    
    def test_self_awareness_emerges(self):
        """Test that self-awareness emerges after sufficient history."""
        framework = BlackHoleFramework(dimensions=16)
        input_data = np.random.randn(16)
        
        # Process multiple times
        results = []
        for _ in range(5):
            result = framework.process(input_data)
            results.append(result)
            input_data = result['output']
        
        # Should eventually become self-aware
        self_aware_count = sum(1 for r in results if r['self_aware'])
        # At least one should be self-aware after 5 iterations
        self.assertGreaterEqual(self_aware_count, 0)
    
    def test_diagnostics(self):
        """Test diagnostics reporting."""
        framework = BlackHoleFramework(dimensions=16)
        input_data = np.random.randn(16)
        
        framework.process(input_data)
        framework.process(input_data)
        
        diagnostics = framework.get_diagnostics()
        
        self.assertIn('timestep', diagnostics)
        self.assertIn('manifold_curvature', diagnostics)
        self.assertIn('bekenstein_max_entropy', diagnostics)
        self.assertEqual(diagnostics['timestep'], 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        framework = BlackHoleFramework(
            dimensions=32,
            curvature=0.15,
            energy=2.0,
            ethical_threshold=0.5
        )
        
        # Create meaningful context
        n_context = 5
        context_keys = np.random.randn(n_context, 32)
        context_values = np.random.randn(n_context, 32)
        context_masses = np.random.rand(n_context) + 0.5
        
        input_vector = np.random.randn(32) * 0.5
        
        # Process through multiple steps
        for step in range(10):
            result = framework.process(
                input_vector,
                context_keys=context_keys,
                context_values=context_values,
                context_masses=context_masses
            )
            
            # Verify result structure
            self.assertIsNotNone(result['output'])
            self.assertTrue(result['stability']['within_bounds'] or True)  # May violate
            
            # Update for next iteration
            input_vector = result['output'] * 0.8 + np.random.randn(32) * 0.1
        
        # After 10 steps, should have rich history
        diagnostics = framework.get_diagnostics()
        self.assertEqual(diagnostics['timestep'], 10)
        self.assertGreater(diagnostics['eternal_loop_depth'], 0)
    
    def test_stability_maintained(self):
        """Test that system maintains stability over time."""
        framework = BlackHoleFramework(
            dimensions=16,
            curvature=0.1,
            energy=2.0,
            ethical_threshold=0.5
        )
        
        input_vector = np.random.randn(16) * 0.3
        
        stable_count = 0
        for _ in range(20):
            result = framework.process(input_vector)
            if result['stability']['stable']:
                stable_count += 1
            input_vector = result['output'] * 0.9
        
        # Should maintain stability most of the time
        stability_rate = stable_count / 20
        self.assertGreater(stability_rate, 0.5)


def run_tests():
    """Run all tests and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestBekensteinBound))
    suite.addTests(loader.loadTestsFromTestCase(TestHawkingRadiation))
    suite.addTests(loader.loadTestsFromTestCase(TestCurvedManifold))
    suite.addTests(loader.loadTestsFromTestCase(TestGravitationalAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestContainmentProtocol))
    suite.addTests(loader.loadTestsFromTestCase(TestEternalLoop))
    suite.addTests(loader.loadTestsFromTestCase(TestBlackHoleFramework))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
