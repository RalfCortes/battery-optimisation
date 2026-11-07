# Status and TODO

Last updated: 2026-11-07. Target: EMA BESS Bidding Algorithm Hackathon. Registration closes **13 Nov 2026 12:00 SGT**, datasets released that afternoon, submission deadline **30 Nov 2026 12:00 SGT**.

## Where we are

| Area | State |
|---|---|
| Market model | NEMS rules: RC383 power groups, RC386 SoC cuts, compliant offers (5+5 storage pairs, 5 per reserve), regulation energy at USEP |
| Forecasting | Chronos-2 zero-shot, re-issued at every gate closure, as-of weather; Chronos-2 for reserve prices; Gaussian copula scenarios |
| Optimisation | SDP water value with Markov prices (11 bins), reserves chosen inside the stage, reserve price states (55 joint states), numba kernels; continuous price state (21 nodes) implemented, backtest running |
| Bidding | Optimal-response curves → band → 5 steps → price-taking → feasibility → conform; reserves at 0 (default) or priced reserve curves (`reserve_offers: curve`, branch `reserves/priced-offers`) |
| Tooling | Saved plans per decision, settlement explorer (`reports/explorer/`, not in git), docs in `docs/` |

12 months, half-hourly re-bidding, energy + reserves (SoC 0–100 MWh, awards up to 100 MW per product):

| Strategy | Profit |
|---|---|
| Perfect foresight | 42.66 M SGD |
| Water value, reserve price states + priced reserve curves | **34.77 M** |
| Water value, point reserve prices + priced reserve curves | 34.62 M |
| Water value, reserve price states | 34.53 M |
| Water value, mean reserve prices | 34.49 M |
| Water value, median reserve prices (default) | 34.22 M |
| Self-schedule | 33.42 M |
| Water value, continuous price state | not run yet (the runs were stopped) |

## Important, before the submission

1. **Hackathon parameters.** Set the SoC range to 5–95 MWh (`config/battery.yaml`) and rerun. The score is net market revenue, so test κ = 0 in the objective.
2. **Ask EMA** (EMA_Energy_Storage@ema.gov.sg):
   - when the algorithm is called (per period or once a day);
   - whether the actual SoC is passed back to it;
   - whether the simulator applies RC386 SoC cuts;
   - what "net market revenue" includes (regulation charges, degradation);
   - whether reserve awards are capped;
   - whether the datasets include the delayed offer stack, reserve/regulation requirements, or AGC utilisation.
3. **Pre-dispatch covariate** as soon as the datasets arrive (EMA provides pre-dispatch information).
4. **Submission packaging.** A single entry point that EMA can run to reproduce the outputs, a dependency list, the algorithm description, and a list of external data sources with licences. Open-Meteo's free API is non-commercial: check its terms or use ECMWF open data. Also check Chronos-2 weights and GPU availability.
5. **Offers for every period.** Produce curves for all future periods (standing offers and fallbacks), not only the next gate closure. If EMA calls once a day, optimise all curves jointly over trajectories.
6. **Offer format.** Add the maximum combined transfer limit and ramp rates (Market Rules 5.2B.2.5–6).
7. **Default reserve model.** The best setting is `reserve_prices: markov` with `reserve_offers: curve` (34.77 M); the default is still point/fixed (34.22 M). Before switching, check how the NEMS engine co-optimises a storage unit's energy and reserve curves.

## Important, method improvements

- **Priced reserve offers: done** as an option (+0.24 to +0.40 M over the year). Next: realistic award caps, because the upside comes from price spikes.
- **Curves against the SoC distribution** at gate closure, instead of the expected SoC.
- **Continuous price state.** Judge it on the running backtests. The tail treatment beyond the 95% quantile is likely needed first, because its top node depends on extrapolation.
- **Realistic reserve caps or price impact.** Today's figures overstate reserve revenue.
- **Regulation near full SoC.** The water value holds 16–17 MW against 30 for the self-schedule. Check whether the cause is the firm-reserve rule or the C4 charge check.
- **Forecast calibration.** The 10–90% band covers 77% instead of 80% (adaptive conformal inference). Check the partial autocorrelation of the errors (is the Markov assumption enough?).

## Data to get (market structure)

EMC data subscription (https://www.home.emcsg.com/services/nems-services/data-subscription; prices incl. GST, as published):

- **Delayed offer stack** (Package A, or Package D "Offer Data" for history at S$2,180 per year):
  - every effective energy, reserve and regulation offer per half-hour period, summed to total MW at each unique price;
  - published D + 28; 16 months online; reserves and regulation from 1 Jul 2019;
  - aggregated, so no unit identity: no price-setter names, but the marginal price level and the supply curve are there.
- **Reserve and regulation requirements, shortfalls, risk adjustment factors, scheduled generation by type** (Package A real-time, 16 months; Package D history from 2003). Forecast schedules (PDS/STS/MOS) are kept 7 days only, so they would have to be archived.
- **Advisory notices** (Package B): energy, reserve and regulation shortfalls, abnormal conditions. These are spike signals.
- Package A is S$1,035.50 per month for 12 months. Ask EMA first whether the hackathon datasets include any of this.

What it would give us:
1. **Residual supply curves:** price as a function of demand (energy) or requirement (reserves). Their slope at the clearing point is the price impact of our MW. That gives realistic reserve awards and prices for a 100 MW battery against the 108 MW regulation requirement, the biggest overstatement today.
2. **A less primitive simulator:** insert our offers into the stack and clear against demand or requirement, so prices react to us instead of being taken as given.
3. **A structural price model:** price = stack⁻¹(demand + reserve needs), with typical stacks by season and hour (stacks change slowly) and forecast demand. It would complement Chronos for spikes. The 28-day delay rules out using live stacks as covariates.
4. **Spike features:** requirements, shortfall advisories and offered headroom (capacity offered minus demand).

Not available anywhere public: unit-level price setters, and regulation (AGC) utilisation data. AGC energy is settled as ordinary metered energy (MEP for injections, USEP for withdrawals); we assume 5% up and 5% down. Ask EMA or PSO.

Already archived ourselves: pre-dispatch energy (USEP, demand, solar, TCL, EHEUR, LCP) and ancillary prices (forecast, plus uncapped reference prices on realised rows). No requirements or offers.

## Later

- Optimise the curves for the two pending intervals jointly.
- LSMC or kernel transitions learnt from trajectories (M6/M9).
- Regime-switching price state, diffusion trajectory models, mixture of forecasters.
- The exact stochastic MILP run, still going (an old experiment, low priority).

## Docs

- `dispatch-methods.html`: methods for dispatch under price trajectories (energy, then reserves, continuous state)
- `forecasting.html`: forecasting layer, today and options
- `bidding.html`: offer construction, rules, gaps
- `pipeline-design.md`: original design doc (4 Nov)
- `overnight-progress-2026-11-05.md`: progress report (5 Nov)
- `deck/project/`: source of the recap deck (slides as HTML sections; the deck itself is at https://claude.ai/artifact/Ti1e6vUwAFycn497kDGug5)
