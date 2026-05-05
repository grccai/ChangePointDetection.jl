import math

from retire.accounts import (Asset, AccountType, Lot, TaxableAccount,
                              TaxAdvantagedAccount, Portfolio)


def test_lot_partial_sale():
    lot = Lot(asset=Asset.STOCK, market_value=10_000, cost_basis=4_000, age_years=2.0)
    gain, sold = lot.sell(2_500)
    assert math.isclose(sold, 2_500)
    # 25% of lot sold; 25% of $6_000 unrealized gain = $1_500
    assert math.isclose(gain, 1_500)
    assert math.isclose(lot.market_value, 7_500)
    assert math.isclose(lot.cost_basis, 3_000)


def test_lot_oversell():
    lot = Lot(asset=Asset.STOCK, market_value=1_000, cost_basis=500, age_years=2.0)
    gain, sold = lot.sell(5_000)
    assert math.isclose(sold, 1_000)
    assert math.isclose(gain, 500)
    assert lot.market_value == 0


def test_taxable_lt_preference():
    tx = TaxableAccount(lots=[
        Lot(Asset.STOCK, 5_000, 1_000, age_years=0.4),  # ST, $4k gain
        Lot(Asset.STOCK, 5_000, 4_000, age_years=2.0),  # LT, $1k gain
    ])
    proceeds, lt, st = tx.sell_for(3_000, Asset.STOCK)
    # Should sell from LT first (lower gain among LT)
    assert math.isclose(proceeds, 3_000)
    assert lt > 0
    assert st == 0


def test_roth_basis_tracking():
    r = TaxAdvantagedAccount(AccountType.ROTH)
    r.deposit(Asset.STOCK, 10_000, is_contribution=True)
    assert r.roth_basis == 10_000
    r.withdraw(3_000)
    assert math.isclose(r.roth_basis, 7_000)


def test_traditional_no_basis():
    t = TaxAdvantagedAccount(AccountType.TRADITIONAL)
    t.deposit(Asset.STOCK, 10_000, is_contribution=True)
    assert t.roth_basis == 0


def test_portfolio_total():
    p = Portfolio()
    p.taxable.lots.append(Lot(Asset.STOCK, 100_000, 60_000, 3.0))
    p.traditional.balances[Asset.BOND] = 50_000
    p.roth.balances[Asset.STOCK] = 25_000
    assert p.total_value() == 175_000
    af = p.asset_fractions()
    assert math.isclose(af[Asset.STOCK], 125_000 / 175_000)
