import numpy as np
import pandas as pd

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def estimate_volatility(returns: pd.Series) -> float:
    returns = returns.dropna() * 100

    if len(returns) < 30:
        return float(returns.std() / 100)

    if ARCH_AVAILABLE:
        try:
            model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
            fitted = model.fit(disp="off")
            forecast = fitted.forecast(horizon=1)
            variance = forecast.variance.iloc[-1, 0]
            return float(np.sqrt(variance) / 100)
        except Exception:
            pass

    return float(returns.std() / 100)