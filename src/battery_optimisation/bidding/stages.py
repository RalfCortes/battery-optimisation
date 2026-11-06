"""Per-issue price stages for the water-value SDP: what each forecast issue says about prices.

A stage source lists the forecast issues (sorted) and, per issue, gives the delivery intervals,
the price states per stage [T, K], their weights [K] and, for Markov prices, the transitions
between states [K, K] (``None``: stagewise independent).

- ``ScenarioStages``: the copula scenarios (forecast run ``scenarios.parquet``), stagewise.
- ``LatticeStages``: Markov price states from the quantile forecasts (``forecast.markov``).
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np
import polars as pl

from battery_optimisation.columns import Col
from battery_optimisation.config import (
    BiddingConfig,
    ForecastConfig,
    MarketConfig,
    MarkovConfig,
    PriceModel,
    Product,
    ReserveMarkovConfig,
    TransitionModel,
)
from battery_optimisation.forecast.markov import FactorLattice, factor_lattice, price_lattice
from battery_optimisation.forecast.walkforward import month_transitions


@dataclass(frozen=True)
class IssueStages:
    stamps: list[datetime]  # delivery intervals [T]
    prices: np.ndarray  # [T, K] price per stage and state (scenario)
    weights: np.ndarray  # [K], sums to 1
    transitions: np.ndarray | None = None  # [K, K] Markov prices; None: independent stages


class StageSource(Protocol):
    @property
    def issues(self) -> list[datetime]: ...  # sorted

    def load(self, issue: datetime) -> IssueStages: ...


def _row_ranges(frame: pl.DataFrame) -> tuple[list[datetime], dict[datetime, tuple[int, int]]]:
    """Issues of a frame sorted by issue time, and each one's row range."""
    issued = frame[Col.ISSUED_AT]
    issues = issued.unique(maintain_order=True).to_list()
    starts = issued.search_sorted(pl.Series(issues), side="left").to_list()
    return issues, dict(
        zip(issues, zip(starts, [*starts[1:], frame.height], strict=True), strict=True)
    )


class ScenarioStages:
    """Stagewise marginals of the copula scenarios (long frame: issued_at, scenario, weight,
    interval_start, usep_forecast)."""

    def __init__(self, scenarios: pl.DataFrame) -> None:
        self._frame = scenarios.sort(Col.ISSUED_AT, Col.SCENARIO, Col.INTERVAL_START)
        self.issues, self._rows = _row_ranges(self._frame)

    def load(self, issue: datetime) -> IssueStages:
        a, b = self._rows[issue]
        part = self._frame.slice(a, b - a)
        scenario_ids = part[Col.SCENARIO].unique(maintain_order=True).to_list()
        wide = part.pivot(on=Col.SCENARIO, index=Col.INTERVAL_START, values=Col.USEP_FORECAST)
        wide = wide.sort(Col.INTERVAL_START)
        weights = part.unique(Col.SCENARIO, maintain_order=True)[Col.WEIGHT].to_numpy()
        return IssueStages(
            stamps=wide[Col.INTERVAL_START].to_list(),
            prices=wide.select([str(k) for k in scenario_ids]).to_numpy(),
            weights=weights.astype(float) / weights.sum(),
        )


class LatticeStages:
    """Markov price states from quantile forecasts (long frame: issued_at, interval_start,
    lead_periods, quantile, usep_forecast). ``rho(first delivery interval)`` gives the lag-1
    correlation of the issue; ``transitions(first delivery interval)`` replaces the Tauchen
    matrix of rho (None: Tauchen)."""

    def __init__(
        self,
        quantiles: pl.DataFrame,
        states: int,
        rho: Callable[[datetime], float],
        transitions: Callable[[datetime], np.ndarray] | None = None,
    ) -> None:
        self._frame = quantiles.sort(Col.ISSUED_AT, Col.LEAD_PERIODS, Col.QUANTILE)
        self._levels = np.sort(self._frame[Col.QUANTILE].unique().to_numpy())
        self.states, self.rho, self.transitions = states, rho, transitions
        self.issues, self._rows = _row_ranges(self._frame)

    def load(self, issue: datetime) -> IssueStages:
        a, b = self._rows[issue]
        part = self._frame.slice(a, b - a)
        q = part[Col.USEP_FORECAST].to_numpy().reshape(-1, self._levels.size)  # [T, Q]
        stamps = part[Col.INTERVAL_START].gather_every(self._levels.size).to_list()
        lattice = price_lattice(self._levels, q, self.states, self.rho(stamps[0]))
        return IssueStages(
            stamps=stamps,
            prices=lattice.prices,
            weights=lattice.weights,
            transitions=(
                lattice.transitions if self.transitions is None else self.transitions(stamps[0])
            ),
        )


def monthly_rho(copula_lag1: dict[date, float]) -> Callable[[datetime], float]:
    """rho of the walk-forward copula in force on a delivery date (latest period start <= it)."""
    return _monthly(copula_lag1, "copula")


def empirical_by_month(
    quantiles: pl.DataFrame,
    realised: pl.DataFrame,
    copula_lag1: dict[date, float],
    forecast: ForecastConfig,
    markov: MarkovConfig,
) -> dict[date, np.ndarray]:
    """Walk-forward empirical transitions for each month of ``copula_lag1`` (its keys), with
    the month's copula rho (or ``markov.rho``) for the Tauchen prior. ``realised`` must cover
    the calibration window before the first month."""
    return {
        month: month_transitions(
            quantiles,
            realised,
            month,
            forecast,
            markov.states,
            markov.rho if markov.rho is not None else rho,
            markov.prior_strength,
        )
        for month, rho in copula_lag1.items()
    }


def _monthly(by_start: dict, what: str) -> Callable[[datetime], object]:
    """Value in force on a delivery date: the latest month start <= it."""
    starts = sorted(by_start)

    def at(first_delivery: datetime):
        i = bisect.bisect_right(starts, first_delivery.date())
        if i == 0:
            raise ValueError(f"no {what} before {first_delivery}")
        return by_start[starts[i - 1]]

    return at


def stage_source(
    scenarios: pl.DataFrame,
    quantiles: pl.DataFrame,
    bidding: BiddingConfig,
    copula_lag1: dict[date, float] | None = None,
    transitions: dict[date, np.ndarray] | None = None,
) -> StageSource:
    """Stages for ``bidding.price_model``: copula scenarios, or the Markov lattice from the
    quantiles with rho from the config, else from the walk-forward copula of each month; with
    ``markov.transitions: empirical`` the month's counted matrix (``transitions``) replaces
    Tauchen."""
    if bidding.price_model is PriceModel.INDEPENDENT:
        return ScenarioStages(scenarios)
    m = bidding.markov
    matrix = None
    if m.transitions is TransitionModel.EMPIRICAL:
        if not transitions:
            raise ValueError("markov.transitions empirical needs the monthly transition matrices")
        matrix = _monthly(transitions, "transition matrix")
    if m.rho is not None:
        fixed = m.rho
        return LatticeStages(quantiles, m.states, lambda _: fixed, matrix)
    if not copula_lag1:
        raise ValueError("Markov prices need markov.rho or the copula lag-1 correlations")
    return LatticeStages(quantiles, m.states, monthly_rho(copula_lag1), matrix)


class ReserveLattices:
    """Reserve price states per decision (``reserve_prices: markov``): the factor lattice of the
    latest ancillary quantile forecast issued at or before the decision (long frame: issued_at,
    interval_start, lead_periods, quantile, usep_forecast, target = price column)."""

    def __init__(
        self,
        quantiles: pl.DataFrame,
        market: MarketConfig,
        cfg: ReserveMarkovConfig,
        products: tuple[Product, ...],
    ) -> None:
        from battery_optimisation.sim.clearing import price_column

        self.cfg, self.products = cfg, products
        self._levels = np.sort(quantiles[Col.QUANTILE].unique().to_numpy())
        self._parts = {}
        for p in products:
            part = quantiles.filter(pl.col(Col.TARGET) == str(price_column(p, market))).sort(
                Col.ISSUED_AT, Col.LEAD_PERIODS, Col.QUANTILE
            )
            if part.is_empty():
                raise ValueError(f"no ancillary quantiles for {p}")
            issues, rows = _row_ranges(part)
            self._parts[p] = (part, issues, rows)

    def load(self, decided_at: datetime, stamps: list[datetime]) -> FactorLattice | None:
        """Lattice over ``stamps`` (None if a product's latest issue does not cover them)."""
        q = {}
        for p in self.products:
            part, issues, rows = self._parts[p]
            i = bisect.bisect_right(issues, decided_at)
            if i == 0:
                return None
            a, b = rows[issues[i - 1]]
            sl = part.slice(a, b - a)
            values = sl[Col.USEP_FORECAST].to_numpy().reshape(-1, self._levels.size)
            at = {
                t: j for j, t in enumerate(sl[Col.INTERVAL_START].gather_every(self._levels.size))
            }
            index = [at.get(t) for t in stamps]
            if any(j is None for j in index):
                return None
            q[p] = values[index]
        return factor_lattice(self._levels, q, self.cfg.loadings, self.cfg.states, self.cfg.rho)
