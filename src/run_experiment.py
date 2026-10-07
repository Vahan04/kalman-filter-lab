"""Generate motion data, run a Kalman filter, and save plots and metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.kalman_filter import KalmanFilter1D
except ModuleNotFoundError:
    from kalman_filter import KalmanFilter1D


def simulate_motion(
    steps: int,
    dt: float,
    initial_position: float,
    velocity: float,
    sensor_noise: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create true motion, clean measurements, and noisy measurements."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if sensor_noise < 0:
        raise ValueError("sensor_noise cannot be negative")

    rng = np.random.default_rng(seed)
    time = np.arange(steps) * dt
    true_position = initial_position + velocity * time
    clean_measurement = true_position.copy()
    noisy_measurement = clean_measurement + rng.normal(0, sensor_noise, steps)
    return {
        "time": time,
        "true_position": true_position,
        "clean_measurement": clean_measurement,
        "noisy_measurement": noisy_measurement,
    }


def run_filter(
    data: dict[str, np.ndarray],
    dt: float,
    measurement_variance: float,
    process_variance: float,
) -> dict[str, np.ndarray]:
    """Estimate position and velocity for every noisy measurement."""
    measurements = data["noisy_measurement"]
    kalman = KalmanFilter1D(
        dt=dt,
        measurement_variance=measurement_variance,
        process_variance=process_variance,
        initial_position=float(measurements[0]),
    )
    estimates = []
    for measurement in measurements:
        kalman.predict()
        estimates.append(kalman.update(float(measurement)))
    states = np.array(estimates)
    data["estimated_position"] = states[:, 0]
    data["estimated_velocity"] = states[:, 1]
    return data


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def save_csv(data: dict[str, np.ndarray], path: Path) -> None:
    fields = list(data)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(zip(*(data[field] for field in fields)))


def save_outputs(data: dict[str, np.ndarray], output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(data, output_dir / "motion_data.csv")

    position_rmse_noisy = rmse(data["true_position"], data["noisy_measurement"])
    position_rmse_filtered = rmse(data["true_position"], data["estimated_position"])
    metrics = {
        "noisy_position_rmse": position_rmse_noisy,
        "filtered_position_rmse": position_rmse_filtered,
        "improvement_percent": 100 * (1 - position_rmse_filtered / position_rmse_noisy),
    }
    (output_dir / "metrics.txt").write_text(
        "\n".join(f"{key}: {value:.6f}" for key, value in metrics.items()) + "\n",
        encoding="utf-8",
    )

    time = data["time"]
    plt.figure(figsize=(11, 6))
    plt.plot(time, data["true_position"], label="True position", linewidth=2)
    plt.plot(time, data["clean_measurement"], "--", label="Clean measurement")
    plt.scatter(
        time,
        data["noisy_measurement"],
        s=14,
        alpha=0.45,
        label="Noisy measurement",
    )
    plt.plot(time, data["estimated_position"], label="Kalman estimate", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.title("Kalman Filter: clean data, noisy data, and filtered estimate")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "kalman_filter_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 4))
    noisy_error = data["noisy_measurement"] - data["true_position"]
    filtered_error = data["estimated_position"] - data["true_position"]
    plt.plot(time, noisy_error, label="Noisy measurement error", alpha=0.7)
    plt.plot(time, filtered_error, label="Filtered error", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Position error")
    plt.title("Position error before and after filtering")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_error_plot.png", dpi=160)
    plt.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--initial-position", type=float, default=0.0)
    parser.add_argument("--velocity", type=float, default=2.0)
    parser.add_argument("--sensor-noise", type=float, default=8.0)
    parser.add_argument("--process-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = simulate_motion(
        args.steps,
        args.dt,
        args.initial_position,
        args.velocity,
        args.sensor_noise,
        args.seed,
    )
    data = run_filter(
        data,
        args.dt,
        measurement_variance=max(args.sensor_noise**2, 1e-12),
        process_variance=args.process_noise,
    )
    metrics = save_outputs(data, args.output_dir)
    print(f"Noisy position RMSE: {metrics['noisy_position_rmse']:.4f}")
    print(f"Filtered position RMSE: {metrics['filtered_position_rmse']:.4f}")
    print(f"Improvement: {metrics['improvement_percent']:.2f}%")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
