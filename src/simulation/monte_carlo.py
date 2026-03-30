import numpy as np


def run_monte_carlo(
    last_price: float,
    predicted_daily_return: float,
    daily_volatility: float,
    days: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    paths = np.zeros((days + 1, n_paths))
    paths[0, :] = last_price

    for path_idx in range(n_paths):
        for day in range(1, days + 1):
            shock = rng.normal(predicted_daily_return, daily_volatility)
            paths[day, path_idx] = paths[day - 1, path_idx] * (1 + shock)

    return paths