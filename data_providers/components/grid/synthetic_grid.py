import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date


def fetch_grid(today: date, seed: int = None) -> pd.DataFrame:
    TARGET_HOURS_PER_DAY = {
        1: 3.50, 2: 3.00, 3: 1.50, 4: 0.50, 5: 0.20, 6: 0.10,
        7: 0.21, 8: 0.25, 9: 0.40, 10: 2.07, 11: 3.12, 12: 6.68,
    }
    BLOCK_DURATION_WEIGHTS = {
        4: 0.35, 8: 0.30, 12: 0.20, 16: 0.10, 24: 0.03, 40: 0.02,
    }
    TIMESTEPS_PER_DAY = 96
    MAX_HOURS_UNTIL_OUTAGE = 24.0

    input_dt = today
    target_dt = input_dt + timedelta(days=1)
    t_month = target_dt.month
    t_day = target_dt.day
    t_dow = target_dt.isoweekday()

    if seed is not None:
        np.random.seed(seed)

    target_h = TARGET_HOURS_PER_DAY.get(t_month, 0.0)
    target_timesteps = int(round(target_h * 4))
    grid = np.ones(TIMESTEPS_PER_DAY, dtype=int)

    if target_timesteps > 0:
        block_sizes = list(BLOCK_DURATION_WEIGHTS.keys())
        block_probs = np.array(list(BLOCK_DURATION_WEIGHTS.values()))
        block_probs /= block_probs.sum()
        filled = 0
        attempts = 0
        while filled < target_timesteps and attempts < 1000:
            attempts += 1
            block_size = np.random.choice(block_sizes, p=block_probs)
            remaining = target_timesteps - filled
            if block_size > remaining: block_size = remaining
            max_start = TIMESTEPS_PER_DAY - block_size
            if max_start <= 0: break
            start = np.random.randint(0, max_start)
            end = start + block_size
            if grid[start:end].sum() == block_size:
                grid[start:end] = 0
                filled += block_size

    outage_remaining_h = np.zeros(TIMESTEPS_PER_DAY)
    next_outage_duration = np.zeros(TIMESTEPS_PER_DAY)
    hours_until_outage = np.full(TIMESTEPS_PER_DAY, MAX_HOURS_UNTIL_OUTAGE)

    outage_blocks = []
    i = 0
    while i < TIMESTEPS_PER_DAY:
        if grid[i] == 0:
            start = i
            while i < TIMESTEPS_PER_DAY and grid[i] == 0: i += 1
            outage_blocks.append((start, i))
        else: i += 1

    for start, end in outage_blocks:
        n = end - start
        for j, idx in enumerate(range(start, end)):
            rem = (n - j) * 0.25
            outage_remaining_h[idx] = rem
            next_outage_duration[idx] = rem

    for t in range(TIMESTEPS_PER_DAY):
        if grid[t] == 0:
            hours_until_outage[t] = 0.0
        else:
            next_outage_start = next((s for s, e in outage_blocks if s > t), None)
            if next_outage_start is not None:
                hours_until_outage[t] = min((next_outage_start - t) * 0.25, MAX_HOURS_UNTIL_OUTAGE)

    for start, end in outage_blocks:
        dur = (end - start) * 0.25
        for t in range(0, start):
            if grid[t] == 1:
                is_nearest = not any(s > t and s < start for s, _ in outage_blocks)
                if is_nearest: next_outage_duration[t] = dur

    df_result = pd.DataFrame({
        'Grid':                 grid,
        'next_outage_duration': np.round(next_outage_duration, 2),
        'outage_remaining_h':   np.round(outage_remaining_h, 2),
        'hours_until_outage':   np.round(hours_until_outage, 2),
    })
    return df_result
