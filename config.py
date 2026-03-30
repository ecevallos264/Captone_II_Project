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


CONFIG = Config()