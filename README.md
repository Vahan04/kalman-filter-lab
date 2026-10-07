# Kalman Filter Lab

A small, reproducible Python experiment showing how a Kalman filter removes
measurement noise from motion data.

The simulated object moves with constant velocity. At every time step:

1. The simulator knows the true position and velocity.
2. A sensor measures position with Gaussian noise.
3. The Kalman filter estimates position and velocity from the noisy sensor.

The project exports the data to CSV and creates plots that compare:

- true motion,
- clean measurements,
- noisy measurements,
- Kalman-filter estimates.

## Model

The state is

```text
x = [position, velocity]
```

For a time step `dt`, the constant-velocity transition model is

```text
F = [[1, dt],
     [0,  1]]
```

The sensor measures position only:

```text
z = Hx + noise,  H = [1, 0]
```

The filter uses the standard prediction and update equations:

```text
x_pred = F x
P_pred = F P F^T + Q
K      = P_pred H^T (H P_pred H^T + R)^-1
x      = x_pred + K(z - H x_pred)
P      = (I - K H) P_pred
```

## Run

From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 src/run_experiment.py --seed 7
```

Outputs are written to `results/`:

- `motion_data.csv`
- `kalman_filter_plot.png`
- `position_error_plot.png`
- `metrics.txt`

The default experiment uses a clean simulated trajectory and a noisy sensor.
You can change the number of steps, sensor noise, and process noise:

```bash
python3 src/run_experiment.py \
  --steps 150 \
  --sensor-noise 8 \
  --process-noise 0.05 \
  --seed 7
```

Run tests with:

```bash
python3 -m pytest
```

## Example result

The filter should have a smaller position RMSE than the noisy measurements.
The exact values depend on the selected random seed and noise settings.
