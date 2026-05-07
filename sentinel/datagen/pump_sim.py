"""
Synthetic centrifugal pump SCADA data generator.

Simulates 6 sensor streams with realistic physics-based relationships.
Injects 4 operating regimes: HEALTHY, CAVITATION, BEARING_WEAR, SEAL_LEAK.

Sensors
-------
- discharge_pressure  [bar]
- suction_pressure    [bar]
- flow_rate           [m3/h]
- motor_current       [A]
- vibration_rms       [mm/s]
- bearing_temp        [C]
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


REGIMES = ["HEALTHY", "CAVITATION", "BEARING_WEAR", "SEAL_LEAK"]
REGIME_LABELS = {r: i for i, r in enumerate(REGIMES)}

SENSOR_COLS = [
    "discharge_pressure",
    "suction_pressure",
    "flow_rate",
    "motor_current",
    "vibration_rms",
    "bearing_temp",
]


@dataclass
class PumpConfig:
    sample_rate_hz: float = 1.0        # 1 sample per second
    duration_hours: float = 100.0      # total sim time
    noise_std: float = 0.02            # fractional noise on all sensors
    seed: int = 42

    # Healthy operating point
    healthy_discharge: float = 6.0     # bar
    healthy_suction: float = 0.8       # bar
    healthy_flow: float = 120.0        # m3/h
    healthy_current: float = 45.0      # A
    healthy_vibration: float = 1.2     # mm/s
    healthy_temp: float = 65.0         # C


def _add_noise(arr: np.ndarray, std_frac: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0, std_frac * np.abs(arr).mean(), size=arr.shape)
    return arr + noise


def _sigmoid_ramp(t: np.ndarray, onset: float, width: float) -> np.ndarray:
    """Smooth 0→1 transition centred at onset over width samples."""
    return 1.0 / (1.0 + np.exp(-(t - onset) / (width / 6)))


def generate_healthy(n: int, cfg: PumpConfig, rng: np.random.Generator) -> np.ndarray:
    """Baseline healthy operation with slow drift and diurnal variation."""
    t = np.arange(n)
    diurnal = 0.01 * np.sin(2 * np.pi * t / (3600 * cfg.sample_rate_hz))

    dp  = cfg.healthy_discharge  * (1 + diurnal + 0.005 * rng.normal(size=n))
    sp  = cfg.healthy_suction    * (1 + diurnal * 0.5 + 0.005 * rng.normal(size=n))
    fl  = cfg.healthy_flow       * (1 - diurnal * 0.3 + 0.005 * rng.normal(size=n))
    cur = cfg.healthy_current    * (1 + diurnal * 0.2 + 0.005 * rng.normal(size=n))
    vib = cfg.healthy_vibration  * (1 + 0.03 * rng.normal(size=n))
    tmp = cfg.healthy_temp       * (1 + 0.005 * rng.normal(size=n))

    sensors = np.stack([dp, sp, fl, cur, vib, tmp], axis=1)
    return _add_noise(sensors, cfg.noise_std, rng)


def generate_cavitation(n: int, cfg: PumpConfig, rng: np.random.Generator,
                         severity: float = 1.0) -> np.ndarray:
    """
    Cavitation: suction pressure drops, discharge pressure drops,
    flow drops, vibration increases sharply, current fluctuates.
    Develops gradually over onset_frac of the segment.
    """
    t = np.arange(n)
    ramp = _sigmoid_ramp(t, n * 0.25, n * 0.3) * severity

    dp  = cfg.healthy_discharge * (1 - 0.18 * ramp + 0.01 * rng.normal(size=n))
    sp  = cfg.healthy_suction   * (1 - 0.45 * ramp + 0.02 * rng.normal(size=n))
    fl  = cfg.healthy_flow      * (1 - 0.25 * ramp + 0.02 * rng.normal(size=n))
    cur = cfg.healthy_current   * (1 + 0.08 * ramp * rng.uniform(0.8, 1.2, n))
    vib = cfg.healthy_vibration * (1 + 2.5  * ramp + 0.1  * np.abs(rng.normal(size=n)))
    tmp = cfg.healthy_temp      * (1 + 0.04 * ramp + 0.005 * rng.normal(size=n))

    sensors = np.stack([dp, sp, fl, cur, vib, tmp], axis=1)
    return _add_noise(sensors, cfg.noise_std, rng)


def generate_bearing_wear(n: int, cfg: PumpConfig, rng: np.random.Generator,
                           severity: float = 1.0) -> np.ndarray:
    """
    Bearing wear: temperature rises steadily, vibration increases
    (with periodic impulses), current slightly elevated.
    Pressure and flow relatively unaffected early on.
    """
    t = np.arange(n)
    ramp = _sigmoid_ramp(t, n * 0.3, n * 0.4) * severity

    # Periodic impulses in vibration (bearing defect frequency)
    bdf_period = int(60 * cfg.sample_rate_hz)
    impulse = 0.8 * severity * np.where((t % bdf_period) < 3, 1.0, 0.0)

    dp  = cfg.healthy_discharge * (1 - 0.03 * ramp + 0.005 * rng.normal(size=n))
    sp  = cfg.healthy_suction   * (1 + 0.02 * ramp + 0.005 * rng.normal(size=n))
    fl  = cfg.healthy_flow      * (1 - 0.05 * ramp + 0.01  * rng.normal(size=n))
    cur = cfg.healthy_current   * (1 + 0.10 * ramp + 0.01  * rng.normal(size=n))
    vib = cfg.healthy_vibration * (1 + 1.8  * ramp + 0.08  * rng.normal(size=n)) + impulse
    tmp = cfg.healthy_temp      * (1 + 0.25 * ramp + 0.005 * rng.normal(size=n))

    sensors = np.stack([dp, sp, fl, cur, vib, tmp], axis=1)
    return _add_noise(sensors, cfg.noise_std, rng)


def generate_seal_leak(n: int, cfg: PumpConfig, rng: np.random.Generator,
                        severity: float = 1.0) -> np.ndarray:
    """
    Seal leak: flow drops (internal recirculation),
    discharge pressure drops, current decreases (less hydraulic load),
    vibration slightly increased.
    """
    t = np.arange(n)
    ramp = _sigmoid_ramp(t, n * 0.2, n * 0.35) * severity

    dp  = cfg.healthy_discharge * (1 - 0.15 * ramp + 0.008 * rng.normal(size=n))
    sp  = cfg.healthy_suction   * (1 + 0.05 * ramp + 0.005 * rng.normal(size=n))
    fl  = cfg.healthy_flow      * (1 - 0.30 * ramp + 0.015 * rng.normal(size=n))
    cur = cfg.healthy_current   * (1 - 0.12 * ramp + 0.01  * rng.normal(size=n))
    vib = cfg.healthy_vibration * (1 + 0.60 * ramp + 0.05  * rng.normal(size=n))
    tmp = cfg.healthy_temp      * (1 + 0.08 * ramp + 0.005 * rng.normal(size=n))

    sensors = np.stack([dp, sp, fl, cur, vib, tmp], axis=1)
    return _add_noise(sensors, cfg.noise_std, rng)


_REGIME_GENERATORS = {
    "HEALTHY":      generate_healthy,
    "CAVITATION":   generate_cavitation,
    "BEARING_WEAR": generate_bearing_wear,
    "SEAL_LEAK":    generate_seal_leak,
}


def simulate(cfg: Optional[PumpConfig] = None) -> pd.DataFrame:
    """
    Generate a full simulation run.

    Returns a DataFrame with columns:
        timestamp, discharge_pressure, suction_pressure, flow_rate,
        motor_current, vibration_rms, bearing_temp, regime, regime_label
    """
    if cfg is None:
        cfg = PumpConfig()

    rng = np.random.default_rng(cfg.seed)
    n_total = int(cfg.duration_hours * 3600 * cfg.sample_rate_hz)

    # Build a schedule: alternate healthy stretches with fault segments
    schedule = []
    remaining = n_total
    segment_min = int(0.5 * 3600 * cfg.sample_rate_hz)   # 30-min min
    segment_max = int(6.0 * 3600 * cfg.sample_rate_hz)   # 6-hr max

    regime_sequence = (
        ["HEALTHY"] * 3
        + ["CAVITATION", "HEALTHY", "BEARING_WEAR", "HEALTHY",
           "SEAL_LEAK", "HEALTHY", "BEARING_WEAR", "CAVITATION",
           "HEALTHY", "SEAL_LEAK", "HEALTHY"]
    )

    for reg in regime_sequence:
        if remaining <= 0:
            break
        hi = min(segment_max, remaining)
        lo = min(segment_min, hi)
        seg_len = int(rng.integers(lo, hi + 1))
        severity = float(rng.uniform(0.6, 1.0))
        schedule.append((reg, seg_len, severity))
        remaining -= seg_len

    if remaining > 0:
        schedule.append(("HEALTHY", remaining, 1.0))

    # Generate each segment
    all_sensors = []
    all_regimes = []

    for reg, length, severity in schedule:
        gen_fn = _REGIME_GENERATORS[reg]
        kwargs = {} if reg == "HEALTHY" else {"severity": severity}
        seg = gen_fn(length, cfg, rng, **kwargs)
        all_sensors.append(seg)
        all_regimes.extend([reg] * length)

    sensors = np.concatenate(all_sensors, axis=0)

    # Clip to physically plausible ranges
    sensors[:, 0] = np.clip(sensors[:, 0], 0.5, 12.0)   # discharge_pressure
    sensors[:, 1] = np.clip(sensors[:, 1], 0.1, 3.0)    # suction_pressure
    sensors[:, 2] = np.clip(sensors[:, 2], 0.0, 250.0)  # flow_rate
    sensors[:, 3] = np.clip(sensors[:, 3], 5.0, 100.0)  # motor_current
    sensors[:, 4] = np.clip(sensors[:, 4], 0.1, 20.0)   # vibration_rms
    sensors[:, 5] = np.clip(sensors[:, 5], 20.0, 120.0) # bearing_temp

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=len(all_regimes),
        freq=f"{int(1/cfg.sample_rate_hz)}s",
    )

    df = pd.DataFrame(sensors, columns=SENSOR_COLS)
    df.insert(0, "timestamp", timestamps)
    df["regime"] = all_regimes
    df["regime_label"] = df["regime"].map(REGIME_LABELS)

    return df


if __name__ == "__main__":
    import os
    cfg = PumpConfig(duration_hours=100, seed=42)
    df = simulate(cfg)
    out = "data/sim/pump_sim_100h.csv"
    os.makedirs("data/sim", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} samples → {out}")
    print(df["regime"].value_counts())
    print(df[SENSOR_COLS].describe().round(3))
