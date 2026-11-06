# Status and TODO

Last updated: 2026-11-06. Target: EMA BESS Bidding Algorithm Hackathon. Registration closes **13 Nov 2026 12:00 SGT**, datasets released that afternoon, submission deadline **30 Nov 2026 12:00 SGT**.

## Where we are

| Area | State |
|---|---|
| Market model | NEMS rules: RC383 power groups, RC386 SoC cuts, compliant offers (5+5 storage pairs, 5 per reserve), regulation energy at USEP |
| Forecasting | Chronos-2 zero-shot, re-issued at every gate closure, as-of weather; Chronos-2 for reserve prices; Gaussian copula scenarios |
| Optimisation | SDP water value with Markov prices (11 bins), reserves chosen inside the stage, reserve price states (55 joint states), numba kernels; continuous price state (21 nodes) implemented, backtest running |
| Bidding | Optimal-response curves → band → 5 steps → price-taking → feasibility → conform; reserves offered at 0 |
| Tooling | Saved plans per decision, settlement explorer (`reports/explorer/`, not in git), docs in `docs/` |

12 months, half-hourly re-bidding, energy + reserves (SoC 0–100 MWh, awards up to 100 MW per product):

| Strategy | Profit |
|---|---|
| Perfect foresight | 42.66 M SGD |
| Water value, reserve price states | **34.53 M** |
| Water value, mean reserve prices | 34.49 M |
| Water value, median reserve prices (default) | 34.22 M |
| Self-schedule | 33.42 M |
| Water value, continuous price state | running |

## Important, before the submission

1. **Hackathon parameters.** Set the SoC range to 5–95 MWh (`config/battery.yaml`) and rerun. The score is net market revenue, so test κ = 0 in the objective.
2. **Ask EMA** (EMA_Energy_Storage@ema.gov.sg):
   - when the algorithm is called (per period or once a day);
   - whether the actual SoC is passed back to it;
   - whether the simulator applies RC386 SoC cuts;
   - what "net market revenue" includes (regulation charges, degradation);
   - whether reserve awards are capped.
3. **Pre-dispatch covariate** as soon as the datasets arrive (EMA provides pre-dispatch information).
4. **Submission packaging.** A single entry point that EMA can run to reproduce the outputs, a dependency list, the algorithm description, and a list of external data sources with licences. Open-Meteo's free API is non-commercial: check its terms or use ECMWF open data. Also check Chronos-2 weights and GPU availability.
5. **Offers for every period.** Produce curves for all future periods (standing offers and fallbacks), not only the next gate closure. If EMA calls once a day, optimise all curves jointly over trajectories.
6. **Offer format.** Add the maximum combined transfer limit and ramp rates (Market Rules 5.2B.2.5–6).
7. **Default reserve model.** Choose between `reserve_prices: markov` (34.53 M) and the median point baseline (34.22 M). The recommendation is markov.

## Important, method improvements

- **Priced reserve offers.** Offer a rising curve priced at opportunity cost instead of a fixed MW at 0 (details in CLAUDE.md, "Important TODOs").
- **Curves against the SoC distribution** at gate closure, instead of the expected SoC.
- **Continuous price state.** Judge it on the running backtests. The tail treatment beyond the 95% quantile is likely needed first, because its top node depends on extrapolation.
- **Realistic reserve caps or price impact.** Today's figures overstate reserve revenue.
- **Regulation near full SoC.** The water value holds 16–17 MW against 30 for the self-schedule. Check whether the cause is the firm-reserve rule or the C4 charge check.
- **Forecast calibration.** The 10–90% band covers 77% instead of 80% (adaptive conformal inference). Check the partial autocorrelation of the errors (is the Markov assumption enough?).

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
