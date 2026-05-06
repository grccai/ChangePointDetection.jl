import numpy as np
import pytest

from retire.accounts import Asset
from retire.returns import (AssetParams, MarketModel, sample_gbm_paths,
                             sample_inflation, _arith_to_log)


def test_arith_to_log_roundtrip():
    mu, sigma = 0.07, 0.18
    ml, sl = _arith_to_log(mu, sigma)
    # Lognormal: E[exp(X)] = exp(mu_log + 0.5 sd_log^2) = 1 + mu_arith
    expected = np.log1p(mu)
    assert abs((ml + 0.5 * sl ** 2) - expected) < 1e-12


def test_gbm_means_match():
    rng_paths = 50_000
    model = MarketModel(
        params={Asset.STOCK: AssetParams(0.06, 0.18),
                Asset.BOND:  AssetParams(0.02, 0.06),
                Asset.CASH:  AssetParams(0.005, 0.01)},
        correlation=np.eye(3),
    )
    r = sample_gbm_paths(model, n_years=1, n_paths=rng_paths, seed=0)
    # Sample mean of arithmetic returns should be near specified arithmetic mean
    assert abs(r[Asset.STOCK].mean() - 0.06) < 0.01
    assert abs(r[Asset.BOND].mean() - 0.02) < 0.005
    assert abs(r[Asset.STOCK].std() - 0.18) < 0.02


def test_correlation_invalid():
    bad = np.array([[1.0, 1.5, 0.0],
                    [1.5, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])  # not PSD
    with pytest.raises(ValueError):
        MarketModel(
            params={Asset.STOCK: AssetParams(0.06, 0.18),
                    Asset.BOND:  AssetParams(0.02, 0.06),
                    Asset.CASH:  AssetParams(0.005, 0.01)},
            correlation=bad,
        )


def test_inflation_deterministic():
    model = MarketModel(
        params={Asset.STOCK: AssetParams(0.06, 0.18),
                Asset.BOND:  AssetParams(0.02, 0.06),
                Asset.CASH:  AssetParams(0.005, 0.01)},
        correlation=np.eye(3),
        inflation_mean=0.03, inflation_vol=0.0,
    )
    inf = sample_inflation(model, 5, 100, seed=0)
    assert np.allclose(inf, 0.03)


def test_sample_deterministic_paths_returns_geometric():
    """Deterministic mode uses geometric mean (CAGR), not arithmetic, so the
    deterministic terminal wealth aligns with the GBM *median* path."""
    from retire.returns import sample_deterministic_paths
    model = MarketModel(
        params={Asset.STOCK: AssetParams(0.06, 0.18),
                Asset.BOND:  AssetParams(0.02, 0.06),
                Asset.CASH:  AssetParams(0.005, 0.01)},
        correlation=np.eye(3),
    )
    out = sample_deterministic_paths(model, n_years=5, n_paths=3)
    # CAGR = (1+mu) / sqrt(1 + sd^2/(1+mu)^2) - 1
    expected_stock = 1.06 / np.sqrt(1 + 0.18**2 / 1.06**2) - 1
    expected_bond = 1.02 / np.sqrt(1 + 0.06**2 / 1.02**2) - 1
    expected_cash = 1.005 / np.sqrt(1 + 0.01**2 / 1.005**2) - 1
    assert np.allclose(out[Asset.STOCK], expected_stock)
    assert np.allclose(out[Asset.BOND], expected_bond)
    assert np.allclose(out[Asset.CASH], expected_cash)
    # Stocks: ~4.4% CAGR (vs 6% arithmetic)
    assert 0.040 < expected_stock < 0.050


def test_sample_historical_paths_dimensions_and_range():
    from retire.returns import sample_historical_paths
    rets, infl = sample_historical_paths(n_years=30, n_paths=100, seed=0)
    for a in (Asset.STOCK, Asset.BOND, Asset.CASH):
        assert rets[a].shape == (100, 30)
        # Real returns should fall within historical extremes
        assert (rets[a].min() > -0.6) and (rets[a].max() < 0.7)
    assert infl.shape == (100, 30)
    # CPI bounds: deflation ~ -10% (1932), inflation ~ +18% (1946)
    assert (infl.min() > -0.12) and (infl.max() < 0.20)


def test_historical_data_loads():
    from retire import historical_data as hd
    yrs, s, b, c = hd.real_returns()
    assert len(yrs) >= 90 and len(yrs) == len(s) == len(b) == len(c)
    # Geometric average real returns over the full period (sanity: stocks
    # ~5-7%, bonds ~1-3%, cash ~0-1%).
    def cagr(arr):
        return float(np.prod(1 + arr) ** (1 / len(arr)) - 1)
    assert 0.04 < cagr(s) < 0.09, f"stock real CAGR {cagr(s):.3f}"
    assert 0.00 < cagr(b) < 0.04, f"bond real CAGR  {cagr(b):.3f}"
    assert -0.005 < cagr(c) < 0.02, f"cash real CAGR {cagr(c):.3f}"
