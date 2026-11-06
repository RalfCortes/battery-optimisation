# Overnight progress — BESS bidding pipeline

Nov 5, 2026 · @Ralf

## Summary

Water-value bidding now clearly works. Giving the SDP persistent (Markov) price states instead of independent ones lifts energy-only profit to **8.69 M SGD a year** with half-hourly forecasts (52% of perfect foresight). The previous best was 6.98 M. Ancillary services remain the biggest lever: up to 25.6 M a year (half-hourly self-schedule) under the current, unconfirmed market assumptions; perfect foresight is 35.9 M with them and 16.7 M without.

- **Energy only:** water value with Markov prices makes 8.69 M re-planned every half-hour (+45% over the rolling self-schedule) and 7.88 M on daily forecasts (+53%). Re-forecasting now helps water value as it should, and the gain holds on two earlier years (+15% and +51%), where the old version lost to the self-schedule.
- **Why the old version stalled:** independent price stages invent spreads. Its SDP expected 25.0 M a year, more than perfect foresight (16.1 M), and re-planning more often made it worse.
- **With ancillary services:** on daily forecasts water value now beats the self-schedule (24.53 M vs 24.04 M, +2.0%). With half-hourly re-forecasts the self-schedule is best overall at 25.59 M, and water value trails at 24.90 M (-2.7%).
- **Weather:** the 18Z backfill is complete (522 of 522 runs), and the intraday run was redone without fallbacks.
- **State of the code:** committed to `main` (latest `cde0193`; Markov prices in `d50cfec`, firm reserves by default in `eb185fa`). 190 tests pass, ruff is clean, and CLAUDE.md is updated.

All figures cover 12 months (2025-10-28 to 2026-10-27) with a 2-interval gate closure, a 100 MW / 100 MWh battery and 85% round-trip efficiency.

## Water value with Markov price states

The water-value SDP treated each half-hour's price as an independent draw. That invents spreads: a high draw now and a low draw an hour later look tradeable, but real prices persist. Each stage now has 11 price states built from the quantile forecast, and a high price makes high prices more likely next (lag-1 correlation from the walk-forward copula, about 0.77).

| Water value, energy only (12 months, gate closure 2) | Daily forecasts | Forecast every half-hour |
| --- | --- | --- |
| Independent price stages (before) | 6.98 M | 6.74 M |
| **Markov price states (now the default)** | **7.88 M** | **8.69 M** |
| Self-schedule, Chronos-2 median | 5.16 M | 5.99 M |
| Perfect foresight | 16.08 M | 16.65 M |

- **Diagnosis.** The old SDP expected 25.0 M SGD over the year and realised 6.98 M; perfect foresight makes 16.1 M. The Markov SDP expects 14.0 M: no longer above what is possible.
- **Rolling paradox explained.** With independent stages, profit fell as re-planning got more frequent: 6.99 M (every 48 intervals), 6.88 M (12), 6.84 M (4), 6.74 M (1), with more cycling. A sharp near stage next to wide independent later stages made every re-plan buy for imaginary spikes.
- **Ruled out:** the end-of-horizon SOC rule (ending each plan at midnight was worse, 6.64 M), the price band (it covers 92% of prices at 1 h lead), and sampling noise alone. The copula scenarios add 16.5 SGD/MWh of noise between consecutive issues against 12.0 for the quantiles, mostly from the k-medoids reduction; the stagewise SDP on the raw quantiles made 7.27 M daily but 6.40 M rolling.
- **Robust settings.** The lag-1 correlation matters but not finely: 0.77 to 0.90 all give 7.7 to 7.9 M daily, while 0 gives 7.27 M and 0.95 gives 6.89 M. 7, 11, 15 and 21 states and price-band margins of 0 to 400 SGD/MWh stay within ±0.1 M. A separate correlation per lead (from the copula, 0.53 to 0.85) gave 7.81 M, no better. Defaults: 11 states, one walk-forward copula correlation per month, no margin.
- **How offers are built.** The value after an interval now depends on that interval's own price, so the curve is the optimal response computed on a price grid between the states, then compressed to 10 tranches as before. The price sweep gap within the forecast band fell from 42 to 24 MW (daily).
- **Noise between runs.** Re-running the half-hourly forecast with complete weather barely changed its accuracy (MAE 45.9 vs 46.2 at 1-1.5 h), yet the rolling self-schedule moved from 6.32 M to 5.99 M. Differences of a few percent between runs are noise; the Markov gain (+0.9 M daily, +1.5 M rolling on identical forecasts) is not.

**It holds on earlier years**, including before the 2023 price cap (daily forecasts, no weather covariate since the weather archive starts July 2025):

| 12 months | Self-schedule | Water value, independent | Water value, Markov |
| --- | --- | --- | --- |
| Jul 2022 - Jun 2023 (before the price cap) | 11.66 M | 8.14 M (-30%) | **13.41 M (+15%)** |
| Oct 2023 - Oct 2024 (after the cap) | 5.69 M | 5.35 M (-6%) | **8.61 M (+51%)** |
| Oct 2025 - Oct 2026 (main test, with weather) | 5.16 M | 6.98 M (+35%) | **7.88 M (+53%)** |

The old water value lost to the self-schedule in both earlier years; its +35% this year was luck. The Markov version wins in all three, on the same forecasts.

## Intraday re-forecasting

Forecasts are issued at every gate closure: 21,840 Chronos-2 issues over 12 months, each with the weather run available at that moment. Re-planning on the latest forecast now helps both strategies: the self-schedule by 16% (5.99 M vs 5.16 M) and water value by 10% (8.69 M vs 7.88 M, table above).

Error of the median forecast by true horizon (MAE, SGD/MWh, 12 months):

| Time from issue to delivery | 1-1.5 h (first deliverable interval) | 2 h | 3 h | 5 h | 7 h | 13 h | 19 h | 25 h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE | 46.2 | 48.4 | 51.3 | 53.4 | 54.5 | 54.8 | 55.2 | 56.9 |

- Intraday weather: the 18Z backfill is complete, and the run was redone with no fallback runs. Accuracy barely changed (CRPS 43.5, 77.0% inside the 10-90% band, 82% of spikes caught).
- Persistence self-schedule: 1.65 M.
- Bugs fixed on the way: crashes on issues cut by a month boundary, the plan-vs-simulation check under rolling plans, and blank weather runs. `forecast run` now saves quantiles before evaluating, and `--reuse-quantiles` restarts from them.

## Chronos-2 vs EMC pre-dispatch

On the first night of archived pre-dispatch runs, Chronos-2 had half the error of EMC pre-dispatch (MAE 22.7 vs 49.2 SGD/MWh), but the sample is tiny: 64 forecast points from 14 runs, 4 Nov 22:30 to 5 Nov 03:30, calm overnight prices (208 to 290 SGD/MWh).

| Forecast | MAE (SGD/MWh) | Bias (SGD/MWh) |
| --- | --- | --- |
| EMC pre-dispatch | 49.2 | -49.2 |
| Chronos-2 median | 22.7 | +1.5 |

- EMC is better one half-hour ahead (MAE 22 vs 32): it already has the current dispatch run. From two half-hours on, Chronos is ahead.
- Every EMC point was below the realised price. A bias that systematic is learnable, so pre-dispatch is a strong candidate covariate for Chronos once the archive holds a few weeks of runs.
- Chronos was issued at each run's half-hour with data before it only; its 10–90% band covered 88% of points. All 11 issues used fallback (older) weather runs.
- To redo once the archive covers at least a week, including the daytime price peaks.

## Probabilistic SOC feasibility

The probabilistic SOC bound is much worse than the worst-case rule, so worst-case stays the default. With bids fixed one hour ahead (gate closure), the bidder caps each offer by the SOC it could have after its still-pending offers clear.

| Feasibility rule | Profit (SGD, 12 months) | Shortfall | Penalty (SGD) |
| --- | --- | --- | --- |
| Worst-case (default) | 6.98 M | 0 MWh | 0 |
| Probabilistic, 1% quantile, 20% margin | 4.64 M | 488 MWh | 2.2 M |
| Probabilistic, 1% quantile, 5% margin | -0.61 M | 1,630 MWh | 7.3 M |
| Probabilistic, 5% quantile, 10% margin | -18.1 M | 5,538 MWh | 24.9 M |
| Probabilistic, 5% quantile, 5% margin | -27.8 M | 7,695 MWh | 34.6 M |
| Probabilistic, 5% quantile, 0% margin | -39.5 M | 10,281 MWh | 46.3 M |

- How it works: the SOC after pending offers is evaluated over each offer's scenario prices; the 5% / 95% quantiles, widened by the margin, replace the worst case. When cleared energy cannot be delivered, the simulator delivers what it can and settles the rest at the 4,500 SGD/MWh penalty.
- Why it fails: shortfalls are frequent, not rare. Pending intervals are treated as independent and the copula scenarios are slightly too narrow, so the quantiles understate how much can clear. Even before penalties, profit is lower (6.8 M vs 6.98 M).
- A larger margin helps but converges to the worst case. The 10% margin and 1% quantile runs did not complete (config changed mid-run) and will be rerun.
- Data: daily Chronos-2 forecasts, test year 28 Oct 2025 to 27 Oct 2026, offers fixed two intervals ahead.

## Ancillary services

With regulation and reserves, the battery's value more than doubles. Perfect foresight rises from 16.7 M to 36.3 M SGD a year, and a forecast-driven self-schedule earns 24.0 M, against 7.0 M for the best energy-only strategy. These are upper bounds until the market assumptions below are confirmed.

| Strategy (12 months, daily forecasts, gate closure 2) | Products | Profit (SGD) | of which ancillary | Cycles/day |
| --- | --- | --- | --- | --- |
| Perfect foresight, full year | energy + ancillary | 36.32 M | 22.73 M | 1.31 |
| Perfect foresight, daily | energy + ancillary | 35.85 M | 22.71 M | 1.30 |
| **Water-value bids, Markov prices, firm reserves (new default)** | energy + ancillary | **24.53 M** | - | 0.48 |
| **Water-value bids, Markov prices, hybrid reserves** | energy + ancillary | **24.39 M** | 20.76 M | 0.56 |
| **Self-schedule, Chronos-2 energy + ancillary prices** | energy + ancillary | **24.04 M** | 21.46 M | 0.49 |
| Water-value bids, independent prices, hybrid reserves | energy + ancillary | 23.80 M | - | 0.56 |
| Water-value bids, independent prices, firm reserves | energy + ancillary | 23.62 M | 20.67 M | 0.44 |
| Self-schedule, ancillary prices by persistence | energy + ancillary | 23.02 M | 20.94 M | 0.68 |
| Perfect foresight, full year | energy only | 16.65 M | - | 2.22 |
| Water-value bids | energy only | 6.98 M | - | 1.25 |
| Self-schedule, Chronos-2 median | energy only | 5.16 M | - | 1.22 |

- The self-schedule captures 94% of the perfect-foresight ancillary revenue (21.46 M of 22.71 M); the gap to perfect foresight is mostly energy (2.98 M vs 13.79 M).
- **With Markov prices, water-value bidding now beats the self-schedule with ancillary services (+2.0% with firm reserves, +1.5% hybrid).** It earns 3.6 M from energy against 2.6 M for the self-schedule, but 0.7 M less from reserves (20.76 M vs 21.46 M): reserves still follow the MILP plan, and the energy curves take room from them.
- Two fixes tonight narrowed that gap from -2.7% to -1.0%:
  - Energy bids at t now keep the SOC that the next interval's planned reserves need.
  - In hybrid mode (`firm_reserves: false`), the charge a plan relies on to hold more than 100 MW of raise reserves is committed as a price-taking bid. With firm reserves, raise plus discharge must instead fit within 100 MW.
- Implemented: typed ancillary offers, pay-as-clear awards, a sustainability check every interval (raises if a reserve cannot be held), energy-reserve co-optimisation in the MILP, reserve limits inside the water-value SDP, and Chronos-2 forecasts of ancillary prices (`forecast ancillary`).

**Half-hourly re-forecasting with ancillary services** (Chronos-2 ancillary forecasts for all 21,840 issues; MAE 9.3 regulation, 8.7 primary, 12.4 contingency):

| Strategy, rolling, energy + ancillary | Profit | Energy | Reserves |
| --- | --- | --- | --- |
| Perfect foresight | 36.32 M | 13.58 M | 22.73 M |
| **Self-schedule, Chronos-2** | **25.59 M** | 3.28 M | 22.31 M |
| Water value, Markov, firm reserves | 24.90 M | 4.06 M | 20.84 M |

- Water value earns 0.8 M more from energy but 1.5 M less from reserves. It offers 99% of the reserves its own plan asks for, so the loss is in the plan, not in gate-closure scaling.
- Four variants did not close the gap: protecting the next two intervals' reserve energy (24.97 M), planning reserves on the median price (24.70 M), hybrid reserves (24.53 M), and both together (24.55 M).
- Closing it needs the SDP to choose reserves itself, per price state, instead of following a separate MILP plan (next steps).

Assumptions to confirm (all in `config/market.yaml`, marked TODO):

- Price-taking with up to 100 MW per product. Singapore's regulation and reserve requirements are small; a 100 MW battery would likely move these prices.
- Reserve prices of group A (the highest-priced of groups A to E).
- Regulation holds 0.25 h of energy both ways per MW, with 10% of its MW actually moved (wear).

## Forecast upgrades

Two changes are kept: Chronos-2 forecasts of the ancillary prices, and a smaller copula shrinkage. Both were judged on downstream profit, not forecast scores alone.

**Chronos-2 for ancillary prices** (kept). The same model and covariates now forecast the regulation and reserve prices, which were previously forecast by persistence:

| Product (group A) | MAE, Chronos-2 median | MAE, persistence |
| --- | --- | --- |
| Regulation | 9.6 | 12.0 |
| Primary reserve | 9.1 | 11.1 |
| Contingency reserve | 12.8 | 15.8 |

The self-schedule with ancillary services goes from 23.02 M to **24.04 M SGD** (+4.4%). New command: `forecast ancillary` adds these forecasts to a forecast run, and `bidding run --ancillary` uses them.

**Copula shrinkage** (kept at 0.02, was 0.1):

| Shrinkage | Lag-1 correlation of scenarios | Water-value profit |
| --- | --- | --- |
| 0.0 | 0.844 | 6.99 M |
| 0.02 | 0.827 | 6.95 M |
| 0.05 | 0.802 | 6.95 M |
| 0.1 | 0.760 | 6.98 M |

Realised lag-1 correlation is 0.833. 0.02 is the closest value that still regularises, and profit hardly moves (±0.03 M), so this matters for the CLAUDE.md invariant more than for money.

**Weather data fixes:**

- A few ECMWF runs came back blank (for example 2026-06-10 and 2026-06-22 18Z). Radiation and precipitation are also missing at lead 0 for every run.
- Blank runs are now dropped and reported, and the as-of rule falls back to the latest usable older run. The full-year runs started before this fix, so they still include these blank runs; the effect is small.
- The 18Z backfill is complete: all 522 runs downloaded after Open-Meteo's daily quota reset at 00:15 UTC. The half-hourly forecast run was redone without fallbacks.

Not tried yet: comparing weather providers, pre-dispatch as a covariate (the archive is only days old), and fine-tuning.

## Decisions and next steps

**Decisions needed from you:**

1. **Ancillary market rules.** These drive most of the value (24 M vs 7 M SGD): award caps per product, which reserve group applies, regulation energy and utilisation, and whether we are price-takers. They are all TODO in `config/market.yaml`. If the challenge caps awards near Singapore's actual requirements, the ranking of strategies may change.
2. **Main strategy.** Energy only: water value with Markov prices and half-hourly forecasts (8.69 M, +45% over the self-schedule). With ancillary services: the half-hourly self-schedule (25.59 M), ahead of water value (24.90 M) until reserves move into the SDP. Which products does the challenge score?
3. **Install `crontab/weather.cron`** (daily 06:17). Weather is not archived live yet; only the pre-dispatch cron is installed.
4. **Open-Meteo licence:** the free API is non-commercial only. Confirm that the challenge use is allowed or plan for a paid key.

**Next steps I would take:**

- **Reserves inside the Markov SDP.** Let the SDP choose the reserve split per price state instead of following the MILP plan. Water value currently gives up 0.7 M of reserve revenue to earn 1.0 M more from energy.
- **Pre-dispatch as a covariate**, once the archive covers a few weeks. It is the market's own forecast and is likely to help short horizons most.
- **Fine-tune Chronos-2** on NEMS history. Keep it only if capture rate improves.
- **Full walk-forward pre-/post-2023 with every strategy.** Done tonight for water value on two earlier years (table in the Markov section). Still to do: the ancillary products on those years, and a weather covariate once older weather is archived.
