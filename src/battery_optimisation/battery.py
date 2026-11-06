"""Battery physics and state: the single place where efficiencies, degradation and limits live.

Convention: powers in MW at the grid connection, ``dt`` in hours, SOC in MWh.
  SOC after an interval = SOC + charge_gain(dt) * charge_mw - discharge_loss(dt) * discharge_mw
  degradation (SGD)     = kappa * (charge_mw + discharge_mw) * dt
The MILP builds its constraints from the same ``charge_gain`` / ``discharge_loss`` coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

from battery_optimisation.config import BatteryConfig

TOL = 1e-6  # MW / MWh tolerance for solver noise


class LimitBreachError(RuntimeError):
    """A requested step would take the battery outside its power or SOC limits."""


@dataclass(frozen=True)
class Step:
    """Outcome of one interval."""

    charge_mw: float
    discharge_mw: float
    soc_mwh: float  # at the end of the interval
    degradation_sgd: float


class Battery:
    def __init__(self, config: BatteryConfig, soc_mwh: float | None = None) -> None:
        self.config = config
        soc = config.soc_init if soc_mwh is None else soc_mwh
        if not config.soc_min - TOL <= soc <= config.soc_max + TOL:
            raise ValueError(f"soc {soc} outside [{config.soc_min}, {config.soc_max}]")
        self._soc = soc

    @property
    def soc_mwh(self) -> float:
        return self._soc

    # --- coefficients (shared with the MILP)

    def charge_gain(self, dt: float) -> float:
        """MWh of SOC gained per MW charged for ``dt`` hours."""
        return self.config.eta_c * dt

    def discharge_loss(self, dt: float) -> float:
        """MWh of SOC lost per MW discharged for ``dt`` hours."""
        return dt / self.config.eta_d

    def degradation_sgd(self, charge_mw: float, discharge_mw: float, dt: float) -> float:
        return self.config.kappa_sgd_per_mwh * (charge_mw + discharge_mw) * dt

    # --- headroom from the current state

    def max_charge_mw(self, dt: float) -> float:
        return min(self.config.p_max_mw, (self.config.soc_max - self._soc) / self.charge_gain(dt))

    def max_discharge_mw(self, dt: float) -> float:
        return min(
            self.config.p_max_mw, (self._soc - self.config.soc_min) / self.discharge_loss(dt)
        )

    # --- state update

    def soc_after(self, charge_mw: float, discharge_mw: float, dt: float) -> float:
        return self._soc + self.charge_gain(dt) * charge_mw - self.discharge_loss(dt) * discharge_mw

    def deliverable(self, charge_mw: float, discharge_mw: float, dt: float) -> tuple[float, float]:
        """The part of a (charge, discharge) request the battery can physically deliver."""
        return (
            max(0.0, min(charge_mw, self.max_charge_mw(dt))),
            max(0.0, min(discharge_mw, self.max_discharge_mw(dt))),
        )

    def regulate(self, up_mwh: float, down_mwh: float) -> float:
        """Apply energy moved by regulation within the interval (grid side: ``up`` injected,
        ``down`` withdrawn). Returns the new SOC; raises ``LimitBreachError`` (state unchanged)
        if it leaves the SOC bounds."""
        cfg = self.config
        soc = self._soc + down_mwh * cfg.eta_c - up_mwh / cfg.eta_d
        if not cfg.soc_min - TOL <= soc <= cfg.soc_max + TOL:
            raise LimitBreachError(f"soc: regulation takes SOC to {soc:.2f} MWh")
        self._soc = min(max(soc, cfg.soc_min), cfg.soc_max)
        return self._soc

    def step(self, charge_mw: float, discharge_mw: float, dt: float) -> Step:
        """Apply one interval. Raises ``LimitBreachError`` (state unchanged) on any breach."""
        cfg = self.config
        if charge_mw > TOL and discharge_mw > TOL:
            raise LimitBreachError(f"both charging ({charge_mw}) and discharging ({discharge_mw})")
        if min(charge_mw, discharge_mw) < -TOL or max(charge_mw, discharge_mw) > cfg.p_max_mw + TOL:
            raise LimitBreachError(
                f"power outside [0, {cfg.p_max_mw}]: charge={charge_mw}, discharge={discharge_mw}"
            )
        soc = self.soc_after(charge_mw, discharge_mw, dt)
        if not cfg.soc_min - TOL <= soc <= cfg.soc_max + TOL:
            raise LimitBreachError(
                f"soc {soc:.3f} MWh outside [{cfg.soc_min}, {cfg.soc_max}] "
                f"(from {self._soc:.3f}, charge={charge_mw:.3f}, discharge={discharge_mw:.3f})"
            )
        self._soc = soc
        return Step(charge_mw, discharge_mw, soc, self.degradation_sgd(charge_mw, discharge_mw, dt))
