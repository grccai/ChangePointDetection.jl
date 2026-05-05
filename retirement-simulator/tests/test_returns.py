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
