"""Return-generating processes for asset classes.

Two models:
  * GBM (correlated geometric Brownian motion) parameterised by *real*
    arithmetic mean and volatility. We sample annual log returns from a
    multivariate normal whose covariance reproduces the requested arithmetic
    moments via mu_log = log(1 + mu_arith) - 0.5*sigma^2.
  * Historical bootstrap (block) — caller supplies a (T, K) array of historical
    annual real returns; we resample T-year paths in blocks of size B with
    replacement. This preserves heavy tails and within-block autocorrelation
    that GBM erases.

Both models return *real* returns (after inflation). Nominal can be recovered
by multiplying by (1+inflation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .accounts import Asset


@dataclass
class AssetParams:
    real_return: float  # arithmetic mean of annual real returns
    vol: float          # annualized stdev of annual real returns


@dataclass
class MarketModel:
    params: dict[Asset, AssetParams]
    # Pearson correlation among annual *log* returns. Order: stock, bond, cash.
    correlation: np.ndarray  # 3x3
    inflation_mean: float = 0.025
    inflation_vol: float = 0.0  # 0 -> deterministic inflation
    yield_fraction: dict[Asset, float] | None = None  # dividend yield as a
    # fraction of total return (taxable account only). Default 0.018 stocks,
    # 1.0 bonds (all return is yield), 1.0 cash.

    def __post_init__(self) -> None:
        if self.yield_fraction is None:
            self.yield_fraction = {Asset.STOCK: 0.018, Asset.BOND: 1.0,
                                   Asset.CASH: 1.0}
        # Validate correlation matrix
        C = np.asarray(self.correlation, dtype=float)
        if C.shape != (3, 3):
            raise ValueError("correlation must be 3x3 (stock, bond, cash)")
        if not np.allclose(C, C.T, atol=1e-9):
            raise ValueError("correlation matrix must be symmetric")
        # PSD check via Cholesky
        try:
            np.linalg.cholesky(C)
        except np.linalg.LinAlgError as e:
            raise ValueError("correlation matrix is not positive semi-definite") from e


def _arith_to_log(mu: float, sigma: float) -> tuple[float, float]:
    """Convert arithmetic mean & stdev of (1+R) to log-return mean & stdev,
    assuming lognormal. Var(log(1+R)) = log(1 + sigma^2 / (1+mu)^2)."""
    var_log = np.log1p((sigma ** 2) / ((1 + mu) ** 2))
    sd_log = np.sqrt(var_log)
    mean_log = np.log1p(mu) - 0.5 * var_log
    return mean_log, sd_log


def sample_gbm_paths(model: MarketModel, n_years: int, n_paths: int,
                     seed: int | None = None) -> dict[Asset, np.ndarray]:
    """Sample annual real returns for stock, bond, cash.

    Returns dict[Asset, ndarray] of shape (n_paths, n_years) with annual
    arithmetic returns (R_t such that V_{t+1} = V_t * (1+R_t))."""
    rng = np.random.default_rng(seed)
    assets = [Asset.STOCK, Asset.BOND, Asset.CASH]
    mu_log = np.empty(3)
    sd_log = np.empty(3)
    for i, a in enumerate(assets):
        ml, sl = _arith_to_log(model.params[a].real_return, model.params[a].vol)
        mu_log[i] = ml
        sd_log[i] = sl
    # Build covariance from correlation and stdevs
    D = np.diag(sd_log)
    cov = D @ model.correlation @ D
    # Sample
    z = rng.multivariate_normal(mean=mu_log, cov=cov, size=(n_paths, n_years))
    # z shape (n_paths, n_years, 3) -> arithmetic
    arith = np.exp(z) - 1.0
    return {a: arith[..., i] for i, a in enumerate(assets)}


def sample_inflation(model: MarketModel, n_years: int, n_paths: int,
                     seed: int | None = None) -> np.ndarray:
    """(n_paths, n_years) array of annual inflation rates."""
    if model.inflation_vol <= 0:
        return np.full((n_paths, n_years), model.inflation_mean)
    rng = np.random.default_rng(seed)
    return rng.normal(model.inflation_mean, model.inflation_vol,
                      size=(n_paths, n_years))


def block_bootstrap(history: np.ndarray, n_years: int, n_paths: int,
                    block_size: int = 5,
                    seed: int | None = None) -> np.ndarray:
    """Block-bootstrap resampling.

    history : (T, K) array of historical observations
    Returns : (n_paths, n_years, K) bootstrap sample.
    """
    T, K = history.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_paths, n_years, K))
    for p in range(n_paths):
        t = 0
        while t < n_years:
            start = rng.integers(0, T - block_size + 1)
            blk = history[start:start + block_size]
            take = min(block_size, n_years - t)
            out[p, t:t + take] = blk[:take]
            t += take
    return out
