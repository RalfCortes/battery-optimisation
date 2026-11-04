# CLAUDE.md — BESS bidding pipeline for Singapore (NEMS)

## Project goal
Build a pipeline that submits price–quantity offers for a battery (BESS) into Singapore's
real-time wholesale market (NEMS) for energy, regulation, primary and contingency reserve.
Success metric: **capture rate** = realised profit / perfect-foresight profit, measured by a
clearing simulator on held-out data.

## Market facts the code must respect
- Real-time gross pool only: no day-ahead market. Pre-dispatch forecasts are non-binding.
- 30-minute dispatch/pricing intervals, ex-ante. 48 intervals per day. Δ = 0.5 h.
- Pay-as-clear. Use USEP (Uniform Singapore Energy Price) as the energy price; ignore nodal spreads.
- Products co-optimised by the market engine: energy, regulation, primary reserve, contingency reserve.
- Reserve sustain times (drive SOC headroom):
  - primary: respond in 9 s, sustain ~10 min → hold (1/6 h × MW) of energy
  - contingency: respond in 10 min, sustain 30 min → hold (1/2 h × MW) of energy
- Offers have a limited number of price–quantity tranches per product (K, configurable).
- Prices have a regime break around the 2023 temporary price cap. Never train or validate across it blindly.
- Units everywhere: MW, MWh, SGD/MWh. Timestamps in Asia/Singapore, interval-start convention.

## Architecture (three layers + simulator)
1. **Forecasting** (`src/forecast/`)
    - Chronos-2 with covariates (core, not optional): EMC pre-dispatch prices/demand, PSO/TSO system data
  (demand, generation outage plans, intertie status), weather (temperature, irradiance), fuel index, regime flag.
   - Outputs marginal quantiles per interval for USEP, regulation, primary, contingency prices.
   - `scenarios.py`: Gaussian copula on PIT values (correlation estimated from past forecast errors)
     → correlated 48-step paths → k-medoids reduction to 20–50 scenarios, keeping spike paths.
   - Never sample steps independently (creates fake spreads and overtrading).
   - Reserve activation = historical event frequency; regulation utilisation = empirical distribution.
   - Current implementation (`config/forecast.yaml`):
     - Issued at gate closure (`issue_lead_intervals`) every `issue_every_intervals` (1 = re-forecast
       every half-hour, rolling 48 steps; 48 = daily), context 28 days; Chronos-2 native 21
       quantiles; GPU via `cuda_visible_devices`. Walk-forward: copula for month M fitted only on
       issues realised before M (shrinkage 0.02 matches the realised lag-1 correlation).
     - Covariates (`forecast/covariates.py`): calendar + regime (KNOWN), demand (PAST only: no archived
       demand forecasts yet), weather (KNOWN).
     - **Weather is as-of** (`data/weather.py`): for issue time T, the latest run r with
       r + availability_delay_h <= T (Open-Meteo single runs; only `ecmwf_ifs` has history). Never use
       `previous_dayN` series for backtests: they include runs published after the issue time.
       Raw 3x3 grid values, one covariate per (variable, point); every transform is reported.
       ECMWF IFS runs used: 06Z/18Z (`run_hours_utc`). Intraday issues take weather per issue from
       `WeatherTrajectories`; a missing or blank run falls back to the latest usable older run
       (counted/reported). Downloads are paced and resumable (429 backoff, 5xx skipped).
     - Never use realised future values as covariates (leakage); `build_input` takes data < issue time.
     - CLI: `uv run forecast run` -> `runs/forecast_*/` (quantiles, scenarios, report.json, backtests).
     - Cron: `crontab/predispatch.sh` (every 30 min), `crontab/weather.sh` (daily, all providers).
2. **Optimisation** (`src/optim/`)
   - `milp.py`: standard battery MILP, used for the baseline and the perfect-foresight benchmark.
     - Separate charge/discharge variables, η_c and η_d, SOC bounds, power limits, terminal SOC value.
     - **Always include one binary per interval forbidding simultaneous charge and discharge.**
     - Reserve/regulation headroom on both power and SOC.
     - Linear degradation cost κ per MWh throughput.
     - Solver: HiGHS via `highspy` modelling API (no pandas/xarray). A full year (17,520 binaries)
       solves in ~5 s.
   - `sdp.py`: main engine. Backward SDP over a SOC grid (50–100 levels) and price scenarios
     → value function V_t(SOC) and its slope (water value). One net action per stage;
     ancillary allocation solved inside the stage.
3. **Bidding** (`src/bidding/`)
   - `water_value.py`: tranche price = slope of V_{t+1} over the SOC range the tranche spans,
     divided by η_d, plus κ (discharge); times η_c, minus κ (charge).
   - `compress.py`: compress the curve to K tranches, breakpoints where the slope is steepest.
   - `feasibility.py`: cap tranche sizes using worst-case SOC at gate closure.
   - `ancillary.py`: reserve/regulation offer price = expected activation cost + degradation
     + value of SOC headroom held.
   - `sweep.py`: parametric price sweep (re-solve with price at t on a grid) as a sanity check.
   - Baseline strategy: self-schedule the MILP plan at floor/cap prices (`ScheduleStrategy`).
   - Current implementation (`config/bidding.yaml`, `WaterValueStrategy`): one SDP per delivery day
     on the copula scenarios (stagewise marginals, 101 SOC levels, terminal penalty below soc_init =
     same end-of-day rule as the baseline). Per interval, at gate closure: curve from V_{t+1} around
     the *expected* SOC (own pending offers cleared under the scenarios) -> trim to the scenario price
     band (+- margin) -> compress to K -> tranches beyond the band made price-taking (floor/cap,
     certain) -> feasibility from the worst-case SOC (price-taking tranches count as cleared).
     With one exact scenario it reproduces the daily perfect-foresight profit (test).
     With intraday issues: one SDP per issue, the self-schedule re-plans at every decision
     (`Horizon.ROLLING`) from the SOC after its committed offers.
     Feasibility `mode: probabilistic` (SOC quantile + margin, `ShortfallPolicy.PENALISE`) exists
     but loses heavily to worst-case on real data (shortfall penalties): keep worst-case.
     Ancillary (`--ancillary`): per issue, a MILP on mean scenario energy + persistence ancillary
     prices commits reserves; their power/SOC needs become SDP `StageLimits`; at gate closure the
     awards are scaled to the worst-case SOC and offered at 0, energy curves capped to the rest.
     CLI: `uv run bidding run` -> `<forecast run>/bidding_<tag>/` (report.json, comparison.md/.csv,
     sweep.parquet, backtests).
4. **Simulator / backtest** (`src/sim/`)
   - Gate closure (`market.yaml: gate_closure_intervals`, 2 = NEMS ~65 min): the offer for t is fixed
     at the start of t-2 with the SOC and prices known then (`MarketView.decided_at`). Forecasts for
     delivery day D are issued at D 00:00 - `issue_lead_intervals` (same gate). Never let a strategy
     or forecaster use anything published after its decision time.
   - Price-taker clearing, interval by interval, no look-ahead:
     discharge tranche clears if offer ≤ realised price; charge if offer ≥ realised price; same for ancillaries.
   - Apply activations, update SOC, book revenue and degradation, assert no limit breached.
   - Core objects (keep this split when adding products):
     - `Battery` (`battery.py`): physics + SOC state; the only place η, κ and limits live. The MILP
       reads its `charge_gain` / `discharge_loss` coefficients; `step()` raises `LimitBreachError`.
     - `Market` (`sim/clearing.py`): realised prices, `clear(IntervalOffers) -> Cleared`;
       `MarketView` = what is known before an interval (past prices only).
     - `Forecaster` protocol (`forecast/base.py`): `forecast(view, end)`. `PerfectForecaster` is the
       oracle for the benchmark; real models (Chronos, ...) implement the same protocol.
     - `Strategy` protocol (`sim/simulator.py`): `offers(view, soc_mwh) -> IntervalOffers`.
       `ScheduleStrategy(forecaster, ..., horizon)` = MILP plan self-scheduled (`bidding/strategy.py`).
     - `Simulator(market, battery).run(strategy) -> Dispatch`: plain loop, strategy → market →
       battery. `Dispatch` (`sim/dispatch.py`) holds the per-interval frame + metrics.
     - `sim/backtest.py`: runs + `RunSummary` (pydantic), saved to `runs/<name>/`
       (`summary.json`, `dispatch.parquet`); CLI `uv run backtest perfect-foresight`.
   - Ancillary services: `IntervalOffers.ancillary` (AncillaryTranche per product), pay-as-clear
     availability at the `reserve_group` price, awards capped by `max_award_mw`; the simulator
     checks every interval that awards are sustainable (`Battery.check_reserves`). Assumptions
     (sustain of regulation, utilisation, caps, group) are TODO in `market.yaml`.
   - Walk-forward monthly splits; report pre- and post-2023 separately.
   - Heavy runs go through `uv run limited --mem-gb 100 --cpus 8 -- <cmd>` (CPU affinity + RSS
     watchdog; cgroup limits are not enforced on this host). Always `env -u UV_INDEX` for uv.

## Repository layout
```
src/
  data/       # loaders for EMC/EMA data, cleaning, regime flags
  forecast/   # base.py (Forecaster), perfect.py, chronos wrapper, covariates, scenarios.py
  battery.py  # Battery: physics + state
  config.py   # pydantic configs
  columns.py  # Col: every column name
  optim/      # milp.py, sdp.py
  bidding/    # strategy.py, water_value.py, compress.py, feasibility.py, ancillary.py, sweep.py
  sim/        # clearing.py (Market), simulator.py, dispatch.py, backtest.py, metrics.py
  cli/        # thin typer apps (registered in [project.scripts]); no logic
tests/       # synthetic.py: shared synthetic prices, forecasters, strategies
config/      # battery.yaml, market.yaml (repo root)
runs/        # backtest outputs (git-ignored)
notebooks/    # exploration only; no logic that the pipeline depends on
data/         # raw/ and processed/ (git-ignored)
```

## Configuration
- `config/battery.yaml`: p_max_mw, e_max_mwh, soc_min, soc_max, eta_c, eta_d, kappa_sgd_per_mwh, soc_init.
- `config/market.yaml`: interval_h (0.5), n_tranches (K), price_floor, price_cap, products enabled,
  reserve sustain hours.
- Never hard-code battery or market parameters in code.

## Conventions
- Python 3.11+, managed with uv. Format with ruff, type-hint public functions.
- **Typed names, never bare strings.** Every categorical identifier is a `StrEnum`: columns
  (`Col.USEP`, `Col.INTERVAL_START` in `columns.py`), data series (`NemsSeries`), feeds
  (`Feed.ENERGY`, `Feed.ANCILLARY`), products (`Product`), etc. Public functions take the enum,
  not `str`. Structured config/records are **pydantic** models (`BatteryConfig`, `MarketConfig` in
  `config.py`; `extra="forbid"`, frozen). Bare strings only for external formats (CSV headers, URLs).
  New column → add it to `Col` first.
- Dataframe processing in **polars** (not pandas). Polars has no index: time series frames carry a
  sorted `interval_start` column (Datetime, tz Asia/Singapore, interval-start convention), first column.
- Store data as parquet (raw downloads included).
- CLIs with **typer**, in `src/battery_optimisation/cli/`, registered in `[project.scripts]`.
- Logging with **structlog** (`structlog.get_logger(__name__)`, event name + key-value fields);
  entry points call `battery_optimisation.logs.configure_logging()`. No `print`/stdlib `logging` in library code.
- Every module has unit tests; the simulator has an end-to-end test on a small synthetic day.
- **TDD** for every new feature or behaviour change: write or update the test first, run it and see
  it fail for the right reason, then implement until it passes. Bug fixes start with a test that
  reproduces the bug. Shared synthetic data helpers live in `tests/synthetic.py`.
- Prefer clear, small functions over frameworks. No deep RL unless explicitly requested.

## Required tests / invariants
- LP: no interval with both charge > 0 and discharge > 0.
- LP: perfect-foresight profit ≥ any strategy's profit on the same data.
- SDP: V_t(SOC) concave in SOC; water value non-increasing in SOC.
- Bidding: discharge offer prices increase with cumulative MW; tranches feasible from worst-case SOC.
- Simulator: SOC always within bounds; reserve awards always sustainable for the full duration.
- Scenarios: empirical lag-1 correlation of sampled paths ≈ that of realised prices.

## Evaluation outputs (`src/sim/metrics.py`)
- Capture rate vs perfect foresight (headline).
- Bidding uplift = pipeline profit − self-schedule profit.
- Revenue split by product; equivalent full cycles per day; degradation cost.
- Forecast diagnostics: CRPS, 10–90% coverage, spike hit rate (top 5% USEP).
  The deciding metric for forecast models is downstream profit, not CRPS.

## Build order
1. Data loaders (USEP, reserve, regulation prices; pre-dispatch forecasts).
2. Clearing simulator + LP perfect-foresight benchmark + self-schedule baseline.
3. Chronos-2 forecasts + copula scenarios → first end-to-end result.
4. SDP water value + offer curves → measure bidding uplift.
5. Forecast upgrades (tail model, fine-tuning), kept only if capture rate improves.

## Open questions (check challenge rules before assuming)
- What is scored: offers through a simulator, or a schedule? Test period?
- Battery parameters, degradation model, cycle limits.
- Which products are allowed; number of tranches; price floor and cap.
- Charging as price-responsive load or scheduled?
- Gate closure and intraday re-bidding allowed?
- Regulation symmetric or split up/down; how activated energy is settled.
- Price-taker assumption valid, or does the simulator model our price impact?

## References
- Pipeline design doc: https://claude.ai/artifact/L7hDFmec2dBYN88VsV9Qaq
- nempy (NEM dispatch, simulator template): https://github.com/UNSW-CEEM/nempy
- Bolun Xu code (SDP arbitrage, storage bidder): https://bolunxu.github.io/codes
- Chronos-2: https://github.com/amazon-science/chronos-forecasting