import numpy as np

from src.kalman_filter import KalmanFilter1D
from src.run_experiment import run_filter, simulate_motion


def test_filter_tracks_clean_constant_velocity_motion():
    data = simulate_motion(80, 0.1, 3.0, 2.0, 0.0, 7)
    result = run_filter(data, 0.1, 1e-12, 0.01)
    assert np.max(np.abs(result["estimated_position"][-20:] - data["true_position"][-20:])) < 1e-5
    assert abs(result["estimated_velocity"][-1] - 2.0) < 1e-5


def test_filter_reduces_noisy_position_error():
    data = simulate_motion(100, 0.1, 0.0, 2.0, 8.0, 7)
    result = run_filter(data, 0.1, 64.0, 0.05)
    noisy_error = np.sqrt(np.mean((data["noisy_measurement"] - data["true_position"]) ** 2))
    filtered_error = np.sqrt(np.mean((result["estimated_position"] - data["true_position"]) ** 2))
    assert filtered_error < noisy_error


def test_invalid_filter_parameters_are_rejected():
    try:
        KalmanFilter1D(0.0, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid dt should raise ValueError")
