"""
Tests for Physical-Layer Covert THz Communications
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from covert_thz import THzTransceiver, CovertnessAnalyzer, SimulationRunner


class TestTHzTransceiver:
    def test_transceiver_creation(self):
        trx = THzTransceiver()
        assert trx.carrier_freq == 0.14e12

    def test_phase_noise(self):
        trx = THzTransceiver()
        noise = trx.generate_phase_noise(1000)
        assert len(noise) == 1000

    def test_embed_covert(self):
        trx = THzTransceiver()
        cover = np.random.randn(100) * 0.1
        bits = np.random.randint(0, 2, 10)
        covert = trx.embed_covert_signal(cover, bits)
        assert len(covert) == 100


class TestCovertnessAnalyzer:
    def test_analyzer_creation(self):
        analyzer = CovertnessAnalyzer()
        assert analyzer.kl_threshold == 0.1

    def test_entropy(self):
        analyzer = CovertnessAnalyzer()
        signal = np.random.randn(1000)
        entropy = analyzer.compute_entropy(signal)
        assert entropy > 0

    def test_kl_divergence(self):
        analyzer = CovertnessAnalyzer()
        normal = np.random.randn(1000) * 0.1
        covert = np.random.randn(1000) * 0.1 + 0.5
        kl = analyzer.compute_kl_divergence(normal, covert)
        assert kl >= 0


class TestSimulationRunner:
    def test_runner_creation(self):
        runner = SimulationRunner()
        assert runner.transceiver is not None

    def test_experiment_run(self):
        runner = SimulationRunner()
        results = runner.run_experiment()
        assert "avg_kl_divergence" in results
        assert "avg_covertness_score" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
