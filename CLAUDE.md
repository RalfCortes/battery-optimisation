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
2. **Optimisation** (`src/optim/`)
   - `lp.py`: standard battery LP/MILP, used for the baseline and the perfect-foresight benchmark.
     - Separate charge/discharge variables, η_c and η_d, SOC bounds, power limits, terminal SOC value.
     - **Always include one binary per interval forbidding simultaneous charge and discharge.**
     - Reserve/regulation headroom on both power and SOC.
     - Linear degradation cost κ per MWh throughput.
     - Solver: HiGHS via linopy (or Pyomo). Size is tiny; expect ms solve times.
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
   - Baseline strategy: self-schedule the LP plan at floor/cap prices.
4. **Simulator / backtest** (`src/sim/`)
   - Price-taker clearing, interval by interval, no look-ahead:
     discharge tranche clears if offer ≤ realised price; charge if offer ≥ realised price; same for ancillaries.
   - Apply activations, update SOC, book revenue and degradation, assert no limit breached.
   - Walk-forward monthly splits; report pre- and post-2023 separately.

## Repository layout
```
src/
  data/       # loaders for EMC/EMA data, cleaning, regime flags
  forecast/   # chronos wrapper, covariates, scenarios.py
  optim/      # lp.py, sdp.py
  bidding/    # water_value.py, compress.py, feasibility.py, ancillary.py, sweep.py
  sim/        # clearing.py, backtest.py, metrics.py
  cli/        # thin typer apps (registered in [project.scripts]); no logic
  config/     # battery.yaml, market.yaml, run.yaml
tests/
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
- Dataframe processing in **polars** (not pandas). Polars has no index: time series frames carry a
  sorted `interval_start` column (Datetime, tz Asia/Singapore, interval-start convention), first column.
- Store data as parquet (raw downloads included).
- CLIs with **typer**, in `src/battery_optimisation/cli/`, registered in `[project.scripts]`.
- Logging with **structlog** (`structlog.get_logger(__name__)`, event name + key-value fields);
  entry points call `battery_optimisation.logs.configure_logging()`. No `print`/stdlib `logging` in library code.
- Every module has unit tests; the simulator has an end-to-end test on a small synthetic day.
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