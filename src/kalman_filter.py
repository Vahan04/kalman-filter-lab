"""A constant-velocity Kalman filter for one-dimensional motion."""

from __future__ import annotations

import numpy as np


class KalmanFilter1D:
    """Estimate position and velocity from noisy position measurements."""

    def __init__(
        self,
        dt: float,
        measurement_variance: float,
        process_variance: float,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if measurement_variance <= 0 or process_variance <= 0:
            raise ValueError("noise variances must be positive")

        self.f = np.array([[1.0, dt], [0.0, 1.0]])
        self.h = np.array([[1.0, 0.0]])
        self.q = process_variance * np.array(
            [[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]
        )
        self.r = measurement_variance
        self.x = np.array([initial_position, initial_velocity], dtype=float)
        self.p = np.eye(2)

    def predict(self) -> np.ndarray:
        """Predict the next state before receiving a measurement."""
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        return self.x.copy()

    def update(self, measurement: float) -> np.ndarray:
        """Correct the prediction using one measured position."""
        innovation = measurement - float((self.h @ self.x)[0])
        innovation_variance = float((self.h @ self.p @ self.h.T)[0, 0] + self.r)
        gain = self.p @ self.h.T / innovation_variance
        self.x = self.x + gain[:, 0] * innovation
        identity = np.eye(2)
        self.p = (identity - gain @ self.h) @ self.p
        self.p = (self.p + self.p.T) / 2
        return self.x.copy()
