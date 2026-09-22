# The sampling ladder, both sigmas

Measured on HAI, PDL1 L=80, `n_step=400`, official runtime, PXDesign donor
`b075867bae942dc0`. Raw per-step table in `event_sigma_ladder.csv`;
regenerate with `scripts/utilities/dump_event_ladder.py`.

`select_event` chooses on the **churned** sigma `t_hat`, not the scheduled
`c_tau`. With `gamma0=1.0, gamma_min=0.01` the churn factor is 2x at every
step except the last, where `c_tau` finally drops below `gamma_min`:

| step | scheduled | churned | gamma |
|---:|---:|---:|---:|
| 0 | 2544.0000 | 5088.0000 | 1.0 |
| 100 | 496.0000 | 992.0000 | 1.0 |
| 200 | 55.7500 | 111.5000 | 1.0 |
| 300 | 2.3750 | 4.7500 | 1.0 |
| 350 | 0.2227 | 0.4453 | 1.0 |
| 390 | 0.0150 | 0.0299 | 1.0 |
| 399 | 0.0068 | 0.0068 | 0.0 |

401 levels; churned sigma spans 0.0068 to 5088. Because the factor is 2x
essentially everywhere, the two families look interchangeable and are not:
the cached-backbone matrix selects by scheduled sigma (0.4355, actual
0.8711), the integrated path by churned sigma (0.429). `--event-sigma 0.429`
resolves to **step 350 of 400** -- 87.5% of the way down.

## Remaining trajectory after the event

U03 alone (no adapter, no feedback), one cell per sigma, same target, length
and generation seed.

| event step | fraction | churned sigma | `ev->fin` (A) |
|---:|---:|---:|---:|
| 99 | 25% | 992.0 | 6.79 |
| 199 | 50% | 111.5 | 7.32 |
| 299 | 75% | 4.75 | **2.15** |
| 350 | 87.5% | 0.4453 | 0.20 |

Two things follow. An event at step 299 leaves ~2.15 A of trajectory, an
order of magnitude more than the current setting and inside the 2-5 A band
worth aiming at. And the top of the ladder **saturates** -- moving from step
199 to step 99 *reduces* remaining displacement -- so there is nothing to buy
above ~step 200 and the usable range is roughly steps 250-330.

## Run-to-run floor

The step-350 cell here and the scored cell are the same configuration and
disagree: `min bb-only` 4.45 A vs 4.14 A, `ev->fin` 0.2010 vs 0.1980. That is
the documented non-reproducibility of PXDesign generation (up to 0.55 A at
identical seed and GPU). Any intervention smaller than ~0.3 A on interface
geometry cannot be resolved by a single unpaired cell.
