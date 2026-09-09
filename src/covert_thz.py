"""
Physical-Layer Covert Communications via Electromagnetic Scattering
and Phase-Noise Inversion in Sub-THz Waveforms

Paper: Physical-Layer Covert Communications in Sub-THz
Venue: ICRCV 2026
Authors: Fernando May et al.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from scipy.special import expit as sigmoid


@dataclass
class THzTransceiver:
    carrier_freq: float = 0.14e12  # 140 GHz
    phase_noise_rms: float = 0.05  # radians
    tx_power: float = 0.1  # Watts

    def generate_phase_noise(self, num_samples: int) -> np.ndarray:
        return np.random.normal(0, self.phase_noise_rms, num_samples)

    def embed_covert_signal(self, cover_signal: np.ndarray,
                             covert_bits: np.ndarray) -> np.ndarray:
        covert_bpsk = 2 * covert_bits - 1
        samples_per_bit = len(cover_signal) // len(covert_bits)
        covert_expanded = np.repeat(covert_bpsk, samples_per_bit)
        covert_signal = covert_expanded * 0.01
        return cover_signal + covert_signal

    def extract_covert_signal(self, received: np.ndarray,
                               phase_noise: np.ndarray) -> np.ndarray:
        demodulated = received - phase_noise * 0.5
        return demodulated


class CovertnessAnalyzer:
    """Analyzes detectability of covert communications."""

    def __init__(self):
        self.kl_threshold = 0.1

    def compute_entropy(self, signal: np.ndarray) -> float:
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))

    def compute_kl_divergence(self, normal: np.ndarray,
                               covert: np.ndarray) -> float:
        normal_hist, edges = np.histogram(normal, bins=50, density=True)
        covert_hist, _ = np.histogram(covert, bins=50, density=True)
        normal_hist = normal_hist + 1e-10
        covert_hist = covert_hist + 1e-10
        normal_hist = normal_hist / normal_hist.sum()
        covert_hist = covert_hist / covert_hist.sum()
        return np.sum(covert_hist * np.log(covert_hist / normal_hist))

    def is_detectable(self, normal: np.ndarray,
                      covert: np.ndarray) -> Dict:
        kl_div = self.compute_kl_divergence(normal, covert)
        entropy_normal = self.compute_entropy(normal)
        entropy_covert = self.compute_entropy(covert)
        return {
            "kl_divergence": kl_div,
            "entropy_normal": entropy_normal,
            "entropy_covert": entropy_covert,
            "is_detectable": kl_div > self.kl_threshold,
            "covertness_score": max(0, 1 - kl_div / self.kl_threshold)
        }


class SimulationRunner:
    """Main simulation for physical-layer covert communications."""

    def __init__(self):
        self.transceiver = THzTransceiver()
        self.analyzer = CovertnessAnalyzer()

    def run_experiment(self) -> Dict:
        num_samples = 10000
        num_trials = 20

        results = {"trials": []}

        for trial in range(num_trials):
            cover = np.random.randn(num_samples) * 0.1
            covert_bits = np.random.randint(0, 2, num_samples // 100)
            phase_noise = self.transceiver.generate_phase_noise(num_samples)

            covert_signal = self.transceiver.embed_covert_signal(cover, covert_bits)
            received = covert_signal + phase_noise

            analysis = self.analyzer.is_detectable(cover, received)
            results["trials"].append(analysis)

        avg_kl = np.mean([t["kl_divergence"] for t in results["trials"]])
        avg_covertness = np.mean([t["covertness_score"] for t in results["trials"]])
        detectable_pct = np.mean([t["is_detectable"] for t in results["trials"]]) * 100

        return {
            "avg_kl_divergence": avg_kl,
            "avg_covertness_score": avg_covertness,
            "detectable_percentage": detectable_pct,
            "total_trials": num_trials
        }


if __name__ == "__main__":
    np.random.seed(20260909)
    print("=" * 60)
    print("Physical-Layer Covert Communications in Sub-THz")
    print("ICRCV 2026 — Simulation Runner")
    print("=" * 60)

    runner = SimulationRunner()
    results = runner.run_experiment()

    print(f"\nAverage KL Divergence: {results['avg_kl_divergence']:.4f}")
    print(f"Average Covertness:    {results['avg_covertness_score']:.4f}")
    print(f"Detectable:            {results['detectable_percentage']:.1f}%")
