# FAA outage attribution forecaster (make.fun polymarket-faa-attribution-01)

Forecasts the Yes price of every market of Polymarket's FAA outage
attribution family at every minute of the next 4 hours, as five quantiles
[0.05, 0.25, 0.5, 0.75, 0.95] per minute.

## Method

- **Median = last price.** At minute scale these prices are a
  near-martingale; every drift term tested made the median worse.
- **Spread = measured, not assumed.** For each anchor horizon
  (1, 5, 15, 60, 240 minutes) the program takes the empirical quantiles of
  that market's h-step price changes over the recent past, centred on their
  own median, and interpolates log-linearly in h. Two trailing windows are
  blended 90/10 (last 6h adaptive, last 24h for tail stability); a market
  short on samples falls back to case-wide pooled quantiles.
- **Recent-activity rescaling.** The spread is multiplied by a per-market
  factor from the last 30 minutes' realised 1-minute volatility relative
  to the last 6 hours, clipped to [0.7, 1.6].

## Practice score

Mean pinball loss **0.276209** on the six practice cases (the relabelled
NVIDIA cut), vs the house persistence baseline's 0.363589 — better on
every case (p00 0.448649, p01 0.312881, p02 0.389911, p03 0.273437,
p04 0.193850, p05 0.105665). Runtime ~0.15 s per case (limit 8 s).

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 entry/main.py
```

Python 3.13, standard library only, offline, deterministic: two runs on
the same case write byte-identical output. Reads every `*.in` case in the
input directory and writes one `<id>.out` per case.
