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
- Storage rules in the market engine (NEMS, since 12 Aug 2025; `market.yaml: storage_rules`,
  `sim/storage_rules.py`; old pooled rules: `config/market_pooled.yaml`):
  - RC383: one storage offer (+ discharge / − charge); power limited **per reserve class**:
    energy + primary + regulation ≤ P and energy + contingency + regulation ≤ P (charging counts
    negative, so reserve can exceed P while charging); regulation keeps energy ± reg within limits.
  - RC386: the engine gets the real SoC 10 min before each period and **cuts the schedule to what
    the SoC allows** (no shortfall). Discharge checks from the start SoC (hours sustained):
    C1 energy ½ + reg ½ + primary ⅙ + contingency ⅓; C2 (energy + reg + primary) ⅙;
    C3 (energy + reg + contingency) ½, all / η_d minus charging × η_c; charge check C4
    (charge + reg) ½ × η_c minus discharge (reserves not counted).
  - Settlement (Market Rules ch. 7): reserves/regulation paid price × scheduled MW (availability);
    energy from activations is ordinary metered energy (injections at MEP, withdrawals at USEP);
    no pay for non-provision (and reserve-group downgrade). The battery also pays regulation charges
    (AFP × gross energy) and a reserve cost share: not modelled yet.
  - Regulation (AGC, one symmetric product = AEMO raise+lower reg) moves energy around the schedule:
    `ancillary.regulation_energy` (up/down share of MW × interval, assumed 0.05/0.05: no public AGC
    data) is settled at USEP, moves the SoC (sim, MILP, SDP value) and adds wear.
    Reserve activations (contingency/primary) are not simulated: availability pays, the energy is
    ordinary energy (≈ break-even); 2025: 73 forced outages, 29 IL contingency activations
    (0.2% of periods) (EMC NEMS Market Report 2025).
  - Market size (EMC 2025): regulation requirement 108 MW, primary ~175 MW, contingency ~411 MW;
    total 2025 spend: reserves 50.9 M, regulation 19.1 M SGD. A 100 MW battery is NOT a price taker
    there: `max_award_mw` (100 each) overstates reserve revenue (TODO: realistic caps / price impact).
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
     - Reserve/regulation headroom on both power and SOC, from `market.storage_rules` (per-class
       power groups, RC386 SoC checks from the start SoC). `firm_reserves`: no credit from the
       opposite energy flow (power or SoC).
     - Linear degradation cost κ per MWh throughput.
     - Solver: HiGHS via `highspy` modelling API (no pandas/xarray). A full year (17,520 binaries)
       solves in ~5 s.
   - `sdp.py`: main engine. Backward SDP over a SOC grid (50–100 levels) and price states
     (Markov transitions, or stagewise-independent scenarios) → value function V_t(SOC) and its
     slope (water value). One net action per stage;
     ancillary allocation solved inside the stage.
3. **Bidding** (`src/bidding/`)
   - `water_value.py`: tranche price = slope of V_{t+1} over the SOC range the tranche spans,
     divided by η_d, plus κ (discharge); times η_c, minus κ (charge). With Markov prices the
     continuation depends on the price itself: curve = optimal response on a price grid.
   - `compress.py`: compress the curve to K tranches, breakpoints where the slope is steepest.
   - `feasibility.py`: cap tranche sizes using worst-case SOC at gate closure.
   - `ancillary.py`: reserve/regulation offer price = expected activation cost + degradation
     + value of SOC headroom held.
   - `sweep.py`: parametric price sweep (re-solve with price at t on a grid) as a sanity check.
   - Baseline strategy: self-schedule the MILP plan at floor/cap prices (`ScheduleStrategy`).
   - Current implementation (`config/bidding.yaml`, `WaterValueStrategy`): one SDP per forecast
     issue (101 SOC levels, terminal penalty below soc_init = same end-of-day rule as the baseline).
     Price stages (`bidding/stages.py`, `price_model`):
     - `markov` (default): `LatticeStages`, 11 equal-probability price states per stage from the
       quantile forecast, AR(1) transitions in normal-score space (`forecast/markov.py`, Tauchen),
       rho = walk-forward copula lag-1 of the month (or `markov.rho`). V per state; offer curve =
       optimal response on a price grid (`water_value.optimal_net_mw`, value after t depends on
       the price at t: `ValueFunction.continuation_at`).
     - `continuous`: same `LatticeStages` with `node_lattice` (`continuous: nodes, z_max`):
       continuous price state (fitted value iteration). V at 21 normal-score nodes on [-3, 3],
       linear in between (hat functions), so E[V(z') | z_j] = sum_l T[j, l] V(z_l) with exact
       hat expectations of the AR(1) (`hat_expectations`): a stochastic matrix, same SDP code.
       Node price = local mean under its hat (keeps the stage mean); AR(1) noise variance
       reduced by rho^2 h^2/6 (what interpolation adds) to keep N(0, 1) stationary. The
       dependence model only sets `transitions` (later: kernel regression on trajectories).
       Methods overview (energy, then reserves, with equations): `docs/dispatch-methods.html`.
     - `independent`: `ScenarioStages`, stagewise marginals of the copula scenarios. Ignores price
       persistence -> fake spreads: the SDP expected 25 M SGD/yr where perfect foresight makes 16 M,
       and re-planning more often made it *worse* (rolling 6.74 M < daily 6.98 M). Markov: daily
       7.77 M, rolling 8.29 M (rho 0.83). Never go back to independent stages.
     Per interval, at gate closure: curve from V_{t+1} around
     the *expected* SOC (own pending offers cleared under the scenarios) -> trim to the scenario price
     band (+- margin) -> compress to K -> tranches beyond the band made price-taking (floor/cap,
     certain) -> feasibility from the worst-case SOC (price-taking tranches count as cleared).
     With one exact scenario it reproduces the daily perfect-foresight profit (test).
     With intraday issues: one SDP per issue, the self-schedule re-plans at every decision
     (`Horizon.ROLLING`) from the SOC after its committed offers.
     Feasibility `mode: probabilistic` (SOC quantile + margin, `ShortfallPolicy.PENALISE`) exists
     but lost heavily to worst-case under the pooled rules (shortfall penalties). Feasibility
     `mode: market`: no SoC caps (with ancillary: planned awards + power caps only), the engine
     cuts to the SoC (RC386).
     Ancillary (`--ancillary`), `reserve_planner`:
     - `milp`: per issue, a MILP on mean stage energy price + Chronos ancillary forecasts
       (`forecast ancillary`; else persistence) commits reserves (firm by default); their power/SOC
       needs become SDP `StageLimits`.
     - `sdp`: the SDP chooses reserves inside each stage: reward R_t(s, s') = best mix at the
       forecast ancillary prices (`optim/reserves.py`: corners of the storage-rule polytope, cached
       per grid), so V values the SOC reserves need. At gate closure: the mix best in expectation
       over the price states (`_choose_reserves`), energy curve = optimal response restricted to
       actions that fit beside it (`offer_curve(allowed=...)`). Markov prices only.
     - `stochastic_milp`: two-stage MILP (`optim.milp.solve_reserve_plan`): reserves shared by
       `stochastic_paths` energy paths sampled from the Markov states (seeded by issue), energy per
       path; reserves -> SDP `StageLimits` as `milp`. HiGHS: ~1.1 s/issue at 20 paths (5+ h per
       half-hourly year); `stochastic_relax` (LP, no binaries, only for this plan whose energy is
       never dispatched) ~0.18 s with the same objective on positive prices.
     At gate closure the awards are scaled to the worst-case SOC (or not: feasibility `market`)
     and offered at 0, energy curves capped to the rest.
     12 months with ancillary, pooled rules (before RC383/386): daily, water value 24.53 M >
     self-schedule 24.04 M; half-hourly, self-schedule 25.59 M > water value 24.90 M (MILP plan).
     NEMS rules + regulation energy, half-hourly (defaults: `reserve_planner: sdp`,
     `feasibility: market`, compliant offers): water value 34.22 M > self-schedule 33.42 M (PF
     42.66 M); MILP planner 32.99 M; stochastic MILP (relaxed, 20 paths) 31.44 M (anticipative
     per-path energy holds too few reserves). Energy only: 9.42 M vs self-schedule 5.99 M (PF 16.65 M).
     These ancillary figures assume up to 100 MW awards per product: unrealistic (market size).
     Offer format (Market Rules ch. 6, enforced by `Market.validate`, applied by `conform`): one
     storage offer of ≤ 5 charge + ≤ 5 discharge pairs in increasing price order (`n_tranches: 5`),
     reserves/regulation ≤ 5 pairs each, prices to 2 decimals, MW to 1 decimal (rounded down);
     volumes ≤ what the expected SoC can deliver. With Markov prices the optimal response can be
     non-monotone (charge at a higher price than a discharge): the curve is the closest rising fit
     (`water_value.rising`, isotonic), and price-taking tranches stop at the other side's prices.
     CLI: `uv run bidding run` -> `<forecast run>/bidding/<tag>/<YYYYmmdd-HHMM>/` (+ `latest` link;
     never overwritten): report.json, comparison.md/.csv, sweep.parquet, decisions.parquet (water
     value trace: issue, stage, decision/expected SoC, price states, planned reserves), backtests/
     (per strategy: summary.json, dispatch.parquet with realised reserve prices, offers.parquet =
     every submitted tranche, plans.parquet = what it planned at each decision, `record_plans`).
     Plans (`strategy.PLAN_COLUMNS`, Float32): self-schedule = its MILP plan; water value = the SDP
     policy (`ValueFunction.policy`, best next SoC per stage/state/SoC) rolled forward from the
     expected SoC through the Markov states (`sdp.expected_schedule`: expected price/power/SoC,
     SoC 10-90%, reserves held via `PairCorners.mixes`). `uv run bidding explore [--run DIR]` ->
     explore.html (data embedded) + explore_plans/ (weekly int16 chunks as base64 JSON, since
     artifacts serve no raw binary: forecast fan of the issue used + both plans per decision,
     fetched on demand; locally `python -m http.server`). Older
     runs: `<forecast run>/bidding_<tag>/`. Notebook 02 reads both.
     `ancillary_price` (median | mean): reserve price point forecast of the water value. Revenue is
     linear in price and reserve prices are right-skewed (realised mean ~14% above the forecast
     median for every product), while energy is valued over its whole distribution: median
     reserves undervalue them. Year (sdp planner): median 34.22 M (+0.80 M vs self-schedule),
     mean 34.49 M (+1.06 M), reserve price states (below) 34.53 M (+1.11 M). Mean/states hold
     more primary/contingency (74 MW) and even less regulation (16-17 MW vs 30 self-schedule).
     `reserve_prices` (point | markov; planner `sdp`): `markov` gives reserve prices their own
     Markov chain, so the SDP state is (energy k, reserve r), 11 x 5 = 55 states. One common
     reserve factor (`forecast.markov.factor_lattice`, `reserve_markov`): z_p = lambda_p F +
     sqrt(1 - lambda_p^2) e_p, F AR(1) rho 0.75 in 5 bins; state price = E[price_p | F bin]
     (revenue is linear, so the conditional mean is exact). Loadings 0.63/0.45/0.78 fit the
     2025-26 forecast-error correlations (reg/pri 0.28, reg/con 0.49, pri/con 0.35; lag-1
     0.71-0.77; vs energy 0.19-0.38, ignored: chains independent). Reserves are still offered at
     0, chosen at gate closure in expectation over the reserve state (`_choose_reserves`); the
     offer curve's continuation is averaged over it. `point` (median or mean) = the baseline.
     `reserve_offers` (fixed | curve; planner `sdp`, feasibility `market`): `fixed` offers the mix
     chosen at the expected prices at 0 SGD/MW/h (always clears). `curve` (`bidding/ancillary.py`)
     keeps that mix at 0 and adds, per product, the MW the same gate-closure choice would hold if
     that product's price were higher (sweep of its price, others at their expected value, over
     every valid mix at the reference SoC), each step priced at its opportunity cost (energy
     value displaced, SoC headroom), <= 5 pairs per product. The energy curve keeps room for the
     planned mix; extra reserves that clear are settled against energy by the engine (surplus,
     RC386). Year: point reserve prices 34.22 -> 34.62 M (+0.40 M, every month up, mostly Sep-Oct
     spikes), reserve states 34.53 -> **34.77 M** (+0.24 M; uplift vs self-schedule 1.35 M,
     +4.0%). Pricing the whole curve instead (planned mix included) lost 0.27 / 0.84 M: below the
     expected price reserves stopped clearing while the energy curve kept their room empty.
     Parallel `limited` runs started together get the same CPUs: stagger them.
     numba kernels (`optim/kernels.py`, joint path and plan mixes only, threads capped to the
     process affinity): 223 -> 58 ms per issue (~17 min per year for the water value).
     Behaviour (12 months, sdp planner, median reserves): the +0.80 M uplift is all Sep-Oct 2026
     (+1.2 M); the other months trail the self-schedule by 20-70 k each, mostly regulation (-1.3 M/yr):
     the water value sits at SoC >= 80 MWh 57% of the time (self-schedule 21%, PF 25%), where the
     RC386 charge check leaves no room for regulation. Engine cuts are mostly small charge top-ups
     at full SoC (expected SoC = mean over states < actual 100 MWh): harmless.
4. **Simulator / backtest** (`src/sim/`)
   - Gate closure (`market.yaml: gate_closure_intervals`, 2 = NEMS ~65 min): the offer for t is fixed
     at the start of t-2 with the SOC and prices known then (`MarketView.decided_at`). Forecasts for
     delivery day D are issued at D 00:00 - `issue_lead_intervals` (same gate). Never let a strategy
     or forecaster use anything published after its decision time.
   - Price-taker clearing, interval by interval, no look-ahead:
     discharge tranche clears if offer ≤ realised price; charge if offer ≥ realised price; same for ancillaries.
   - Then, with `storage_rules.enforce_soc`, the engine cuts the cleared schedule to the storage rules
     at the SoC at the start of the interval (`enforce_soc`): an LP keeping the highest-surplus
     tranches (realised price − offer price per MW: price-taking energy first). Cut MW: `soc_cut_mw`.
   - `shortfall_penalty` (enabled by default, can be disabled): undeliverable cleared energy with
     `ShortfallPolicy.PENALISE`; with SoC enforcement nothing reaches it today (safety net).
   - Apply activations, update SOC, book revenue and degradation, assert no limit breached.
   - Core objects (keep this split when adding products):
     - `Battery` (`battery.py`): physics + SOC state; the only place η, κ and limits live. The MILP
       reads its `charge_gain` / `discharge_loss` coefficients; `step()` raises `LimitBreachError`.
     - `sim/storage_rules.py`: market rules on top of the physics (power per group, SoC checks,
       engine cuts); used by the simulator, the MILP and the water-value strategy.
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
     checks every interval that the schedule meets the storage rules (`check_storage_rules`). Assumptions
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

## Important TODOs
- **Priced reserve offers: done as an option** (`reserve_offers: curve`, branch
  `reserves/priced-offers`; +0.24 to +0.40 M over the year). Still to do: check how the NEMS
  co-optimisation treats a storage unit's energy vs reserve curves (our engine settles conflicts
  by surplus per MW); then decide the default (`curve` with `reserve_prices: markov` is best).
  The upside MW come from price spikes, so realistic award caps / price impact matter even more.
- Realistic reserve award caps / price impact (`max_award_mw` = 100 each overstates revenue).

## Open questions (check challenge rules before assuming)
- What is scored: offers through a simulator, or a schedule? Test period?
- Battery parameters, degradation model, cycle limits.
- Which products are allowed; number of tranches; price floor and cap.
- Charging as price-responsive load or scheduled?
- Gate closure and intraday re-bidding: answered by Market Rules ch. 6 (1 Oct 2025). Storage is
  a generation registered facility and must always hold a valid *standing offer* (by day of week)
  for every dispatch period, for energy and for each reserve/regulation product it is registered
  for (5.1.1-5.1.3). *Offer variations* for specific periods replace it, and both can be revised
  at any time (5.1.4) until 65 min before the period (gate closure, 10.4.1; exceptions are
  outages/advisories only, and the price may not change). The engine uses the last accepted
  valid offer at schedule computation (5.8.2-5.8.3). So re-bidding every half-hour is allowed:
  the offer that counts for t is the one in force at t - 65 min (our rolling design), while
  offers for all later periods must always exist (fallback; they also feed pre-dispatch, 12-36 h,
  and market outlook, 7 days). 5.1.5: offered quantity must not exceed what is reasonably
  expected to be available by more than max(10 MW, 5%). The storage offer also states a maximum
  combined transfer limit (energy + reserve + regulation) and ramp rates (5.2B.2.5-6): not in
  our offer format yet. Hackathon (EMA, Nov 2026): "bid quantities and prices ... for each
  trading period"; when EMA calls the algorithm is not stated (ask EMA).
- Regulation symmetric or split up/down; how activated energy is settled.
- Price-taker assumption valid, or does the simulator model our price impact?

## References
- Pipeline design doc: https://claude.ai/artifact/L7hDFmec2dBYN88VsV9Qaq
- nempy (NEM dispatch, simulator template): https://github.com/UNSW-CEEM/nempy
- Bolun Xu code (SDP arbitrage, storage bidder): https://bolunxu.github.io/codes
- Chronos-2: https://github.com/amazon-science/chronos-forecasting