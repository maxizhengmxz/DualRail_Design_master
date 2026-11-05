"""One group-two link, matching existing K entries and retaining the new channel."""
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.optimize import brentq

WF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WF), str(WF.parent)]
from integrated_readout_range import frozen_model
from seven_readout import SevenReadout


class OffSearch:
    """Shared four-physical-reference spectrum for target and bypass diagnostics."""
    def __init__(self, physical, basis, capture_participation=False):
        self.physical = physical
        self.basis = basis
        self.model = frozen_model(physical)
        self.capture_participation = capture_participation

    @lru_cache(maxsize=128)
    def at(self, frequency):
        system = SevenReadout(self.model, self.physical, float(frequency),
                              basis=self.basis, off_only=True, eigenlevels=48).solve()
        detail = system.physical_off_detail()['physical_exchange']
        if not detail['valid']:
            raise ValueError('Physical single-excitation projection failed overlap screen')
        matrix = np.asarray(detail['effective_hamiltonian_ghz'])
        result = dict(frequency_ghz=float(frequency), physical_exchange=detail,
                    A2_B1_hz=float(matrix[1, 2] * 1e9),
                    A1_B1_hz=float(matrix[0, 2] * 1e9),
                    A1_B2_hz=float(matrix[0, 3] * 1e9),
                    A2_B2_hz=float(matrix[1, 3] * 1e9))
        if self.capture_participation:
            from studies.group_two_idle_participation import participation_detail
            result['state_participation'] = participation_detail(system)
        return result

    def root(self, left, right, channel='A2_B1_hz'):
        root = brentq(lambda f: self.at(float(f))[channel], left, right, xtol=1e-10)
        row = self.at(root)
        if abs(row[channel]) >= 1:
            raise ValueError('Reported OFF root exceeds the 1 Hz within-basis threshold')
        return row

    def local_root(self, seed, bounds):
        for halfwidth in (.0005, .002, .01, .05):
            left, right = max(bounds[0], seed-halfwidth), min(bounds[1], seed+halfwidth)
            if self.at(left)['A2_B1_hz'] * self.at(right)['A2_B1_hz'] <= 0:
                return self.root(left, right)
        return None


