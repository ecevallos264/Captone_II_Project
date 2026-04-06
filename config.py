from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "TSLA", "NVDA"])
    start_date: str = "2019-01-01"
    end_date: str = "2026-01-01"
    sequence_length: int = 30
    test_size: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    forecast_days: int = 30
    monte_carlo_paths: int = 300
    random_seed: int = 42
    use_market_features: bool = True
    market_tickers: List[str] = field(default_factory=lambda: ["^GSPC", "^IXIC"])
    # S&P 500 Direction Prediction (Logistic Regression)
    sp500_ticker: str = "^GSPC"
    sp500_start_date: str = "2005-01-01"
    sp500_end_date: str = "2026-01-01"
    sp500_lag_days: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 10, 20])
    sp500_test_ratio: float = 0.2
    sp500_logistic_C: float = 1.0
    sp500_logistic_max_iter: int = 1000


CONFIG = Config()