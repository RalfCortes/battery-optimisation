# BESS Bidding Pipeline — Singapore (NEMS)

Nov 4, 2026 · @Ralf

## Context and goal

We operate a BESS that must submit price–quantity offers into Singapore's real-time market, every 30 minutes, for energy and ancillary services. The pipeline turns probabilistic forecasts into a schedule, then into offer curves that stay sensible whatever actually clears.

The market is close to Australia's NEM, not to Europe's day-ahead/intraday design ([Australian Energy Council](https://energycouncil.com.au/analysis/singapore-some-familiar-issues-in-an-unfamiliar-context)).

| Feature | NEMS rule | What it means for us |
| --- | --- | --- |
| Market structure | Real-time gross pool only; no day-ahead market; non-binding pre-dispatch | No DA/ID sequencing; we re-decide every interval |
| Interval | 30 min, ex-ante dispatch and pricing | 48 decisions per day |
| Pricing | Pay-as-clear; generators paid nodal prices, load pays USEP (weighted average) | Use USEP as the price; nodal spread is small on the island |
| Products | Energy, regulation, primary reserve, contingency reserve, co-optimised by the market engine | We offer each product; the engine splits our capacity, but our SOC is our problem |
| Reserve sustain times | Primary: respond in 9 s, hold 9 min 51 s. Contingency: respond in 10 min, hold 30 min ([EMA handbook](https://www.ema.gov.sg/content/dam/corporate/resources/educational-materials/handbook/handbook-pdfs/english/EMA-Resources-Educational-Materials-Handbook-Energy-Storage-Systems-20201022.pdf)) | Reserve awards require SOC headroom |
| Price protections | High price cap; vesting contracts; temporary price cap since 2023 | Structural break in historical prices |
| Offer format | Limited number of price–quantity tranches per product | Offer curves must be compressed to K steps |

Working assumptions until the challenge rules say otherwise: we are a price-taker, battery parameters (MW, MWh, efficiency, SOC limits, degradation) are given, and the score is realised profit over a test period.

## Pipeline at a glance

&#91;embedded content: pipeline · 3 layers + market clearing, re-run every 30 min\]

Each layer hands the next one a richer object than a point forecast: scenarios, then a value function, then SOC-contingent offer curves. The loop re-runs every interval with the realised SOC and prices.

## Layer 1 — Upstream forecasting

The forecaster must deliver **joint price scenarios over the next 48 intervals**, not a single curve: battery profit lives in the tails and in the spreads between intervals.

| Target | Model | Why it matters |
| --- | --- | --- |
| USEP, 30 min, 48+ steps | Time-series foundation model (TSFM) | Energy arbitrage; water value |
| Regulation, primary and contingency reserve prices | TSFM | Ancillary revenue |
| Regulation utilisation (net energy per MW awarded) | Empirical distribution | SOC drift, efficiency losses |
| Reserve activation | Historical event frequency (rare, not a TSFM job) | Expected activation energy |

Covariates are core, not optional: EMC pre-dispatch demand and price forecasts (also the baseline to beat), PSO/TSO system data (demand, generation outage plans, intertie status), calendar, temperature and irradiance, fuel index, and a flag for the post-2023 price-cap regime.

| Option | Pros | Cons |
| --- | --- | --- |
| Chronos-2 with weather and TSO covariates (zero-shot first) | No training; native covariate support | Not tuned to Singapore spikes |
| TSFM + tail model (quantile gradient boosting or spike classifier) | Fixes TSFMs' weakness on extreme prices | More moving parts |
| Fine-tuned TSFM on NEMS history | Learns local shape | Overfitting risk across the 2023 regime break |

Chronos-2 ranked first on fev-bench and GIFT-Eval ([Chronos team, 2026](https://www.ecb.europa.eu/press/conferences/shared/pdf/20260323_forecasting/Guerron_presentation.pdf)). On Belgian electricity prices it was the most accurate TSFM with covariates but struggled in the extreme deciles, where an ML ensemble did better ([Belgium study, 2026](https://www.emergentmind.com/papers/2605.17045)). So: start zero-shot, add the tail model once the backtest can measure its profit impact.

**Quantiles to scenarios.** TSFMs give marginal quantiles per step; sampling steps independently creates fake spreads and overtrading. Use a Gaussian copula fitted on past forecast errors to draw correlated 48-step paths, then reduce to 20–50 scenarios, keeping spikes.

**Evaluation.** CRPS and band coverage for diagnostics; the deciding metric is profit when the forecast drives the pipeline.

## Layer 2 — Optimisation engine

The engine is a standard battery LP, not custom work; its real job is to produce the water value V\_t(SOC) that drives bidding.

A plain LP already covers the physics we need:

- Round-trip efficiency, through separate charge and discharge variables with η\_c and η\_d.
- Power and energy limits, SOC bounds, terminal SOC value.
- Reserve and regulation headroom: spare power, plus enough SOC to sustain primary (about 10 min) and contingency (30 min) awards.
- Degradation as a linear cost per MWh of throughput.

We include **one binary per interval** forbidding simultaneous charge and discharge, by default. Without it, the LP will charge and discharge at once whenever burning energy through losses pays: negative prices, SOC stuck at its upper limit, or making SOC room for regulation-down and reserve awards. At 48 intervals the MILP still solves in milliseconds. Beyond that, extras are only needed for efficiency that varies with power or SOC (piecewise approximation) and cycle-depth degradation. The SDP engine is unaffected: it picks one net action per stage on the SOC grid.

| Option | Role | Effort |
| --- | --- | --- |
| Deterministic LP, rolling horizon | Baseline, and perfect-foresight benchmark on realised prices | Low (standard formulation) |
| SDP over a SOC grid, stage problem solved per grid point and scenario | **Main engine**: gives V\_t(SOC) for bidding | Medium |
| Multi-scenario stochastic LP | Optional, only if the SDP stage problem struggles with all four products | Medium |

The SDP state stays one-dimensional (SOC); ancillary products are decided inside each stage, which is small enough to enumerate.

## Layer 3 — Bidding strategy

Each tranche is priced at the **opportunity cost of the energy it uses**: the water value at the SOC level that tranche would take us to. One LP shadow price gives a single point; an offer needs the whole curve over SOC.

| Option | How offer prices are set | Role |
| --- | --- | --- |
| Self-schedule | Offer the LP schedule at the price floor or cap | Baseline |
| Parametric price sweep | Sweep the price at t, re-solve, record dispatch q\_t(p); that step function is the offer | Sanity check against water values |
| Water-value curves | Tranche price = slope of V\_{t+1} over the SOC range it spans, adjusted for efficiency and degradation | **Main strategy** |

For a discharge tranche taking SOC from s to s − q (charge mirrors it with η\_c and −κ):

```latex
p^{\text{dis}} = \frac{1}{\eta_d}\cdot\frac{1}{q}\int_{s-q}^{s} \frac{\partial V_{t+1}}{\partial e}(x)\,dx + \kappa
```

Rules:

1. **Compress to K tranches**, with breakpoints where the water value is steepest.
2. **Stay feasible under any clearing**: size tranches from the worst-case SOC at gate closure.
3. **Reserve and regulation offers**: price = expected activation cost + degradation + value of the SOC headroom held (near zero mid-SOC, high near the limits).
4. **Let the market co-optimise** energy against reserves within an interval; our job is only the cross-interval SOC logic it cannot see.

## Backtesting and evaluation

We judge the pipeline on realised profit from a price-taker clearing simulator, as a share of perfect-foresight profit.

Each interval, in order: build offers from current SOC and forecasts (no look-ahead); clear tranches against the realised price (discharge clears if offer ≤ price, charge if offer ≥ price, same for ancillaries); apply activations; update SOC and book revenue and degradation.

| Benchmark | Measures |
| --- | --- |
| Perfect-foresight LP on realised prices | Upper bound |
| Self-schedule of the median-forecast LP plan | Value of the forecast alone |
| Full pipeline | Final result |

KPIs: **capture rate** (profit / perfect foresight), **bidding uplift** (pipeline − self-schedule), revenue by product, cycles per day, and zero failed reserve deliveries. Use walk-forward monthly splits, reporting pre- and post-2023 separately.

## Open questions and build order

### To confirm in the challenge rules

- [ ] What is scored: offers cleared by a simulator, or a schedule? Over which test period?
- [ ] Battery parameters: MW, MWh, efficiencies, SOC limits, degradation model, cycle limits.
- [ ] Which products we can offer (energy only, or also regulation, primary, contingency).
- [ ] Number of price–quantity tranches per product, price floor and cap.
- [ ] Is charging offered as price-responsive load or scheduled?
- [ ] Gate closure and whether offers can be revised during the day.
- [ ] Regulation definition (symmetric or split up/down) and how activated energy is settled.
- [ ] Data provided: price history, pre-dispatch forecasts, activation data, covariates.
- [ ] Are we a price-taker, or does the simulator include our price impact?

### Build order

1. Data pipeline: USEP, reserve and regulation prices, pre-dispatch forecasts.
2. Clearing simulator plus a standard battery LP for the benchmark and baseline (no custom formulation work).
3. Zero-shot Chronos-2 forecasts and copula scenarios: first end-to-end result with self-scheduling.
4. SDP water value and offer curves: measure bidding uplift.
5. Forecast upgrades (tail model, fine-tuning), kept only if capture rate improves.

## Sources and useful links

### Market rules and context (Singapore)

| Link | What it is | Use for us |
| --- | --- | --- |
| [EMA — Market Rules page](https://ema.gov.sg/regulations-licences/regulations/market-operations/market-rules) | Official entry point to the Singapore Electricity Market Rules and rule-change process | Authoritative rules; points to EMC's rule and manual library |
| [Singapore Electricity Market Rules, 2022 edition (PDF mirror)](https://policy.thinkbluedata.com/sites/default/files/Singapore%20Electricity%20Market%20Rules%20%282022%20Ed.%29.pdf) | Full rules text, chapters 1–8 | Offer format, dispatch, reserve and regulation definitions; check for later amendments |
| [EMA — ESS Vision and Industry Consultation Paper (Oct 2026)](https://www.ema.gov.sg/content/dam/corporate/our-energy-story/energy-grid/EMA-Our-Energy-Story-Energy-Grid-Energy-Storage-Systems-Vision-Industry-Consultation-Paper.pdf.coredownload.pdf) | EMA's current storage plans and planned rule review | Likely context for the challenge |
| [EMA — Handbook for Energy Storage Systems (2020)](https://www.ema.gov.sg/content/dam/corporate/resources/educational-materials/handbook/handbook-pdfs/english/EMA-Resources-Educational-Materials-Handbook-Energy-Storage-Systems-20201022.pdf) | BESS licensing, market participation, reserve definitions | Reserve response and sustain times |
| [EMC — NEMS Market Report 2025](https://www.home.emcsg.com/publications/-/media/Comms/NEMS-Market-Reports/NEMS-Market-Report-2025_Final.pdf) | Annual review of energy, reserve and regulation markets | Price levels, ancillary requirements, regime context |

### Historical data

| Link | What it is | Notes |
| --- | --- | --- |
| [EMC home — NEMS Prices](https://www.home.emcsg.com/) | Public half-hourly USEP and demand pages (NEMS Market Data menu) | Free source for USEP history |
| [EMC — Data subscription](https://www.home.emcsg.com/services/nems-services/data-subscription) | Paid packages: nodal prices, forecast data, offer data, system data | Power market data about S$2,507 per year; offer data about S$2,180 |
| [EMA — Average monthly USEP, 2005–2026](<https://www.ema.gov.sg/content/dam/corporate/resources/statistics/files/pdf/Average%20Monthly%20USEP%20(as%20of%20Sep%2026).pdf.coredownload.pdf>) | Monthly USEP statistics | Sanity checks and regime view |
| [EMC — NEMS Market Report 2024](https://www.home.emcsg.com/-/media/Comms/NEMS-Market-Reports/NEMS-Market-Report-2024_Final1.pdf) | Annual report with reserve and regulation price history | Ancillary price levels by year |

### Code and papers with similar ideas

| Link | What it is | Relevance |
| --- | --- | --- |
| [UNSW-CEEM/nempy](https://github.com/UNSW-CEEM/nempy) | Python model of the Australian NEM dispatch, with co-optimised FCAS | Template for our clearing simulator |
| [UNSW-CEEM/NEMPRO](https://awesome.ecosyste.ms/projects/github.com%2Funsw-ceem%2Fnempro) | Forecast → plan → bid pipeline for NEM portfolios, battery example | Same architecture as ours (bidding module unfinished) |
| [prakaa/NEMStorageUnderUncertainty](https://www.github.com/prakaa/NEMStorageUnderUncertainty) | BESS arbitrage simulations driven by NEM pre-dispatch forecasts | Rolling-horizon backtest design; value of forecast quality |
| [prakaa/nem-rebidding-analysis-2012-2021](https://awesome.ecosyste.ms/projects/github.com%2Fprakaa%2Fnem-rebidding-analysis-2012-2021) | Analysis of real NEM bids, incl. BESS | How real batteries shape offer curves |
| [kim-mskw/bessbidder](https://science.ecosyste.ms/projects/8962) | Open multi-market BESS bidding (MILP, rolling intrinsic, deep RL) | Benchmark structure; European markets |
| [Bolun Xu — code page](https://bolunxu.github.io/codes) | Analytical SDP arbitrage, transferable storage bidder (Python), degradation models | Closest match to our water-value bidding |
| [Zheng & Xu — Analytical SDP for storage arbitrage](https://arxiv.org/pdf/2108.06000v2) | SDP for real-time arbitrage; 50–90% of perfect-foresight profit | Method and benchmark for Layer 2 |
| [Xu — SoC-segment bidding (CAISO talk)](https://bolunxu.github.io/assets/files/Xu_Storage_CAISO.pdf) | Bids built from the value-to-go function; SoC-segment bids beat power bids by 5–10% | Direct support for Layer 3 |
| [Lee & Sun — GPU DP for storage bidding](https://arxiv.org/pdf/2511.15629v1) | Multistage stochastic DP producing full bid curves | Faster SDP and multi-segment bids |
| [Seifert et al. — SDDP for battery reserve and spot trading](https://arxiv.org/pdf/2406.08390) | SDDP across day-ahead, intraday and reserve | If we extend to multi-product SDP |
| [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | Chronos-2 code and models | Layer 1 model; native past and future covariate support ([Chronos-2 report](https://arxiv.org/pdf/2510.15821v1)) |
| [gschivley/battery\_model](https://www.github.com/gschivley/battery_model) | Simple LP battery arbitrage on NYISO prices | Quick standard LP to adapt for the benchmark |
