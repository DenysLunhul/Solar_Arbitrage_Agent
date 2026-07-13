"""Shared synthetic load-profile generator — single source of truth for consumption shapes.

Profiles are fraction-of-peak curves scaled by a configurable peak_kw:
    office    — low nights, 08-18h plateau, weekends at 45% (the legacy enterprise shape)
    two_shift — plateau 06-22h, low overnight, weekends at 75%
    flat      — near-constant 24/7, no weekend dip

Used by both the live pipeline (synthetic_load.fetch_load -> generate_day) and
training-time domain randomization (train.py RandomConfigWrapper -> generate_series).

generate_day() reproduces the legacy synthetic_load.fetch_load() output exactly for
profile='office', peak_kw=60 (same per-date seeding and rng call order).
generate_series() draws from a caller-supplied rng and does not match generate_day's
per-date stream — it exists for fast vectorized generation over dataset-aligned arrays.
"""

from datetime import date

import numpy as np

PROFILES = ('office', 'two_shift', 'flat')
DEFAULT_PEAK_KW = 60.0
DEFAULT_PROFILE = 'office'

_PARAMS = {
    'office':    {'weekend_factor': 0.45, 'noise': (0.88, 1.12), 'spikes': True},
    'two_shift': {'weekend_factor': 0.75, 'noise': (0.90, 1.10), 'spikes': False},
    'flat':      {'weekend_factor': 1.00, 'noise': (0.95, 1.05), 'spikes': False},
}

# legacy office spike band: steps 36-68 of a 96-step day = 09:00-17:00
_SPIKE_HOUR_LO = 9.0
_SPIKE_HOUR_HI = 17.0


def base_fraction(profile: str, t_hours: np.ndarray) -> np.ndarray:
    """Fraction-of-peak shape evaluated at t = hour + minute/60 (vectorized)."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown load profile '{profile}', expected one of {PROFILES}")

    t = np.asarray(t_hours, dtype=np.float64)

    if profile == 'office':
        # legacy base_kw / 60: 20 kW night, ramp 06-08, 60 kW plateau 08-18, ramp 18-20
        frac = np.full_like(t, 20.0 / 60.0)
        up = (t >= 6.0) & (t < 8.0)
        frac[up] = (20.0 + (t[up] - 6.0) / 2.0 * 40.0) / 60.0
        frac[(t >= 8.0) & (t <= 18.0)] = 1.0
        down = (t > 18.0) & (t < 20.0)
        frac[down] = (60.0 - (t[down] - 18.0) / 2.0 * 40.0) / 60.0
        return frac

    if profile == 'two_shift':
        frac = np.full_like(t, 0.35)
        up = (t >= 5.5) & (t < 6.0)
        frac[up] = 0.35 + (t[up] - 5.5) / 0.5 * 0.65
        frac[(t >= 6.0) & (t <= 22.0)] = 1.0
        down = (t > 22.0) & (t < 22.5)
        frac[down] = 1.0 - (t[down] - 22.0) / 0.5 * 0.65
        return frac

    return np.ones_like(t)  # flat


def _weekend_mask(day_of_week: np.ndarray) -> np.ndarray:
    """Weekend detection for both ISO 1-7 (dataset, isoweekday) and 0-6 conventions."""
    dow = np.asarray(day_of_week)
    return dow >= 6 if dow.min() >= 1 else dow >= 5


def generate_day(profile: str, peak_kw: float, target_date: date) -> np.ndarray:
    """96-step (15-min) load for one calendar day in kW, deterministic per date."""
    params = _PARAMS[profile] if profile in PROFILES else None
    if params is None:
        raise ValueError(f"Unknown load profile '{profile}', expected one of {PROFILES}")

    rng = np.random.default_rng(int(target_date.strftime('%Y%m%d')))
    t = np.array([h + m / 60.0 for h in range(24) for m in (0, 15, 30, 45)])
    base = base_fraction(profile, t) * peak_kw

    is_weekend = target_date.isoweekday() >= 6
    if is_weekend:
        base = base * params['weekend_factor']

    load = base * rng.uniform(*params['noise'], size=len(t))

    if params['spikes'] and not is_weekend:
        n_spikes = int(rng.integers(1, 3))
        centers = rng.integers(36, 69, size=n_spikes)
        for center in centers:
            amp = rng.uniform(1.20, 1.35)
            for offset in (-1, 0, 1):
                idx = int(center) + offset
                if 0 <= idx < len(load):
                    load[idx] = min(load[idx] * amp, base[idx] * 1.35)

    return load


def generate_series(profile: str, peak_kw: float, hours: np.ndarray, minutes: np.ndarray,
                    day_of_week: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Load in kW for dataset-aligned Hour/Minute/Day_of_week arrays (vectorized)."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown load profile '{profile}', expected one of {PROFILES}")
    params = _PARAMS[profile]

    t = np.asarray(hours, dtype=np.float64) + np.asarray(minutes, dtype=np.float64) / 60.0
    weekend = _weekend_mask(day_of_week)

    base = base_fraction(profile, t) * peak_kw
    base[weekend] *= params['weekend_factor']
    load = base * rng.uniform(*params['noise'], size=len(t))

    if params['spikes']:
        dow = np.asarray(day_of_week)
        boundaries = np.flatnonzero(np.diff(dow) != 0) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(t)]))
        in_band = (t >= _SPIKE_HOUR_LO) & (t <= _SPIKE_HOUR_HI)
        for s, e in zip(starts, ends):
            if weekend[s]:
                continue
            band = s + np.flatnonzero(in_band[s:e])
            if len(band) == 0:
                continue
            n_spikes = int(rng.integers(1, 3))
            centers = rng.choice(band, size=n_spikes)
            for center in centers:
                amp = float(rng.uniform(1.20, 1.35))
                for offset in (-1, 0, 1):
                    idx = int(center) + offset
                    if s <= idx < e:
                        load[idx] = min(load[idx] * amp, base[idx] * 1.35)

    return load
