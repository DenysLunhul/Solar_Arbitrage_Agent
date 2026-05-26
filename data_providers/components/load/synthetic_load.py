import numpy as np
import pandas as pd
from datetime import date, timedelta

def fetch_load(today: date) -> pd.DataFrame:
    tomorrow = today + timedelta(days=1)
    day_of_week = tomorrow.isoweekday()
    rng = np.random.default_rng(int(tomorrow.strftime('%Y%m%d')))
    steps = [{"Hour": h, "Minute": m} for h in range(24) for m in [0, 15, 30, 45]]
    df = pd.DataFrame(steps)

    def base_kw(hour: int, minute: int) -> float:
        t = hour + minute / 60.0
        if t < 6.0:
            return 20.0
        elif t < 8.0:
            return 20.0 + (t - 6.0) / 2.0 * 40.0
        elif t <= 18.0:
            return 60.0
        elif t < 20.0:
            return 60.0 - (t - 18.0) / 2.0 * 40.0
        else:
            return 20.0

    base = np.array([base_kw(r.Hour, r.Minute) for r in df.itertuples()])

    if day_of_week >= 6:
        base = base * 0.45

    noise = rng.uniform(0.88, 1.12, size=len(df))
    load = base * noise

    if day_of_week < 6:
        n_spikes = int(rng.integers(1, 3))
        centers = rng.integers(36, 69, size=n_spikes)
        for center in centers:
            amp = rng.uniform(1.20, 1.35)
            for offset in (-1, 0, 1):
                idx = int(center) + offset
                if 0 <= idx < len(load):
                    load[idx] = min(load[idx] * amp, base[idx] * 1.35)

    df["Load"] = load.round(2)
    return df[["Load"]]
