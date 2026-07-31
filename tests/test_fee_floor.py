"""The fee algebra is the load-bearing claim of this strategy, so it is tested hardest.

If `s_floor` is wrong, every grid this strategy places is mispriced.
"""

import math

import pytest

from grid.strategy import (
    Config,
    breakeven_spacing,
    net_edge_per_round_trip,
    spacing_floor,
)


def test_breakeven_is_2f_over_1_minus_f_not_2f():
    """The naive 2f figure is BELOW break-even. The 1/(1-f) correction has a sign."""
    fee = 0.0010
    assert breakeven_spacing(fee) == pytest.approx(0.002 / 0.999, rel=1e-12)
    assert breakeven_spacing(fee) == pytest.approx(0.0020020020, abs=1e-9)
    assert breakeven_spacing(fee) > 2 * fee  # strictly worse than the naive number


def test_net_edge_is_exactly_zero_at_breakeven():
    for fee in (0.0002, 0.0005, 0.0010, 0.0020, 0.0075):
        assert net_edge_per_round_trip(breakeven_spacing(fee), fee) == pytest.approx(0.0, abs=1e-15)


def test_net_edge_is_negative_at_the_naive_2f_spacing():
    """A grid spaced at exactly 2f loses on EVERY round trip, forever."""
    fee = 0.0010
    net = net_edge_per_round_trip(2 * fee, fee)
    assert net < 0
    assert net == pytest.approx(-0.0000020, abs=1e-9)


def test_spacing_floor_delivers_exactly_the_requested_edge():
    for fee in (0.0002, 0.0010, 0.0020):
        for edge in (0.0005, 0.0015, 0.0050):
            s = spacing_floor(fee, edge)
            assert net_edge_per_round_trip(s, fee) == pytest.approx(edge, rel=1e-12)


def test_default_config_floor_matches_the_documented_numbers():
    cfg = Config()
    assert cfg.maker_fee == 0.0010
    assert cfg.breakeven == pytest.approx(0.0020020, abs=1e-6)
    assert cfg.s_floor == pytest.approx(0.0035035, abs=1e-6)
    assert net_edge_per_round_trip(cfg.s_floor, cfg.maker_fee) == pytest.approx(0.0015, abs=1e-9)


def test_net_edge_is_monotone_increasing_in_spacing():
    fee = 0.0010
    values = [net_edge_per_round_trip(s, fee) for s in (0.001, 0.005, 0.01, 0.05, 0.1)]
    assert values == sorted(values)


def test_fee_drag_formula_matches_measured_definition():
    """drag(s) = f(2+s)/s, the fraction of gross spread capture eaten by fees."""
    fee = 0.0010
    for s in (0.005, 0.01, 0.0422, 0.08):
        gross = s
        net = net_edge_per_round_trip(s, fee)
        assert (gross - net) / gross == pytest.approx(fee * (2 + s) / s, rel=1e-12)


def test_drag_at_default_spacing_is_under_five_percent():
    """Sanity-check the headline claim that median spacing keeps drag ~5%."""
    fee, s = 0.0010, 0.0422  # measured median daily ATR% on BTC_USDT
    assert fee * (2 + s) / s == pytest.approx(0.0484, abs=0.002)


def test_validate_rejects_a_cap_below_the_floor():
    with pytest.raises(ValueError, match="no spacing is legal"):
        Config(s_cap=0.001).validate()


def test_validate_rejects_a_fee_that_swallows_the_target_edge():
    """At a 10% fee no spacing under the cap can clear the hurdle."""
    with pytest.raises(ValueError):
        Config(maker_fee=0.10, taker_fee=0.10).validate()


def test_dd_kill_is_derived_from_the_inventory_cap():
    """A risk threshold has to be a threshold OF something."""
    assert Config(cap_range=0.50).dd_kill == pytest.approx(0.35)
    assert Config(cap_range=0.30).dd_kill == pytest.approx(0.21)
    assert Config(cap_range=0.20).dd_kill == pytest.approx(0.15)  # clamped at the floor
    assert Config(cap_range=0.90).dd_kill == pytest.approx(0.50)  # clamped at the ceiling
    assert Config(dd_kill_override=0.25).dd_kill == pytest.approx(0.25)


def test_dd_kill_always_exceeds_structurally_normal_drawdown_at_shipped_defaults():
    """The original 20% constant failed this and fired on normal operation."""
    cfg = Config()
    structurally_normal = cfg.cap_range * cfg.asset_max_dd
    assert cfg.dd_kill >= structurally_normal - 1e-9
    assert 0.20 < structurally_normal  # the old hardcoded value was below it


def test_geometric_spacing_gives_equal_net_edge_at_every_level():
    """The reason geometry is geometric: fee is a percentage, so edge must be too."""
    cfg = Config()
    s = 0.03
    anchor = 60_000.0
    levels = [anchor / (1 + s) ** j for j in range(1, cfg.n_levels + 1)]
    edges = [
        (level * (1 + s) * (1 - cfg.maker_fee) - level * (1 + cfg.maker_fee)) / level
        for level in levels
    ]
    assert max(edges) - min(edges) < 1e-12
    assert all(e > 0 for e in edges)


def test_arithmetic_spacing_would_not_have_that_property():
    """Contrast case: with a fixed price step the deepest level earns a different edge.

    A 6-level ladder at a 3% arithmetic step spreads fractional spacing by ~18%
    across the ladder, so a step sized to clear the fee hurdle at the top is a
    different proposition at the bottom. Deeper ladders and wider steps make it
    worse. This is why the fee floor is a per-level check under arithmetic
    spacing but a grid-wide invariant under geometric spacing.
    """
    anchor, step = 60_000.0, 1_800.0  # 3% of the anchor
    levels = [anchor - step * j for j in range(1, 7)]
    fractional = [step / level for level in levels]
    assert max(fractional) / min(fractional) == pytest.approx(1.183, abs=0.01)
    assert not math.isclose(fractional[0], fractional[-1], rel_tol=1e-3)

    # A deeper ladder makes the divergence severe: at 15 levels the bottom rung's
    # fractional spacing is ~1.76x the top rung's.
    deep = [step / (anchor - step * j) for j in range(1, 16)]
    assert max(deep) / min(deep) == pytest.approx(1.76, abs=0.02)
