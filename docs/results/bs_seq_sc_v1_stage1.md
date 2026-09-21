# bs_seq_sc_v1 — §9 stage 1 (reconstruction on the 31 val dimers)

Run `495864`, output `/scratch/m000137-pm06/Proteo-AA/pxf/runs/bs_seq_sc_eval/pilot_v2`.
13 arms (U03 donor + 2 arms × 2 seeds × 3 checkpoints), 31 structures × 4 sigmas,
n = 124 per arm, 0 errors. EMA weights. Selection rule as predeclared in
`configs/bs_seq_sc/selection.yaml`, written before any evaluation existed.

**This is reconstruction NLL, not designability.** Per that config,
`out_of_scope: [designability, binder_success, interface_quality]`.

## Checkpoint selection

Primary = median-over-structures `seq_per_token_nll`. Guardrail = `sc_loss`
no more than 5% worse than that arm's own step 500.

| arm | seq NLL | sc_loss | sc vs own @500 | eligible |
|---|---|---|---|---|
| S03_s0@0500 | 1.53940 | 0.002404 | +0.00% | yes |
| S03_s0@1000 | 1.55837 | 0.002492 | +3.67% | yes |
| S03_s0@2000 | 1.55915 | 0.002566 | +6.74% | **no** |
| S03_s1@0500 | 1.53183 | 0.002434 | +0.00% | yes |
| S03_s1@1000 | 1.54355 | 0.002673 | +9.84% | **no** |
| S03_s1@2000 | 1.55680 | 0.002658 | +9.19% | **no** |
| J03_s0@0500 | 1.50692 | 0.002353 | +0.00% | yes |
| J03_s0@1000 | 1.50623 | 0.002655 | +12.83% | **no** |
| J03_s0@2000 | 1.49853 | 0.002507 | +6.55% | **no** |
| J03_s1@0500 | 1.50951 | 0.002584 | +0.00% | yes |
| J03_s1@1000 | 1.50993 | 0.002570 | −0.56% | yes |
| J03_s1@2000 | 1.50412 | 0.002629 | +1.72% | yes |
| U03 (donor) | 1.50460 | 0.002644 | — | — |

All four arms select **@500**. The guardrail is doing real work rather than
decorating the result: it rules out 6 of 8 later checkpoints, because side-chain
loss *rises* with training in both arms. Note `J03_s0@2000` has the best raw
sequence number of any arm (1.49853) and is ineligible — exactly the trade the
guardrail was declared to refuse.

## Comparisons at the selected checkpoints

Paired per structure, then aggregated. Negative favours the first arm.
95% CI from 10,000 bootstrap resamples; `*` = CI excludes zero.

| comparison | median Δ seq NLL | 95% CI | Wilcoxon p | frac favouring |
|---|---|---|---|---|
| **J03 − S03, seed 0** | **−0.01549** | [−0.02251, −0.01052]* | 9.6e−05 | 0.90 |
| **J03 − S03, seed 1** | **−0.01276** | [−0.02061, −0.00714]* | 4.1e−04 | 0.84 |
| J03 − U03, seed 0 | +0.00373 | [+0.00126, +0.00629]* | 1.1e−05 | 0.10 |
| J03 − U03, seed 1 | +0.00147 | [−0.00126, +0.00415] | 2.2e−01 | 0.42 |
| S03 − U03, seed 0 | +0.02290 | [+0.01266, +0.02734]* | 9.2e−06 | 0.10 |
| S03 − U03, seed 1 | +0.01519 | [+0.00957, +0.02271]* | 7.0e−05 | 0.13 |

Side-chain loss against the donor (negative = adapter packs better):

| comparison | median Δ sc_loss | 95% CI |
|---|---|---|
| J03 − U03, seed 0 | −0.00022676 | [−0.00048921, −0.00013783]* |
| J03 − U03, seed 1 | −0.00014582 | [−0.00027522, −0.00006063]* |
| S03 − U03, seed 0 | −0.00019339 | [−0.00033878, +0.00001418] |
| S03 − U03, seed 1 | −0.00017632 | [−0.00037500, −0.00008408]* |

## Reading

**The primary comparison is positive and holds in both seeds.** With routing
held fixed, adding the sequence objective improves masked-sequence NLL over the
side-chain-only arm by ~0.013–0.015 nats/token, favouring J03 on 84–90% of
structures. That is what J03 was built to test, and it passes.

**But neither arm beats the uncoupled donor on sequence.** S03 is clearly worse
than U03 (+0.015 to +0.023, both seeds, CI excludes zero): the side-chain
objective alone degrades sequence, as expected. J03 recovers most of that
degradation but does not cross back over — it is +0.0037 (seed 0, significant)
and +0.0015 (seed 1, not significant) *worse* than the donor. The honest summary
is that the sequence objective repairs most of the damage the side-chain
objective does, rather than adding sequence skill the donor lacked.

Both arms do improve packing over the donor, in 3 of 4 seed/arm cells.

**One caveat that does not fit the NLL story.** Top-1 recovery (median over
structures) is U03 0.6061, S03_s0 0.6140, S03_s1 0.6009, J03 0.5960 both seeds.
J03 has the *best* NLL of the adapters and the *worst* recovery. NLL rewards
calibration and recovery only rewards the argmax, so the two can diverge, but it
means the J03 − S03 win should not be restated as "J03 recovers more native
sequence" — it does not.

Effect sizes throughout are ~1% of an NLL of ~1.5. They are consistent and
statistically resolvable, not large.

## Provenance note

The first attempt at this evaluation (job 495828) reported n=20 and 26 of 31
structures failing. That was a single bad structure (3u4b) plus CUDA context
poisoning from an async device-side assert; see commit `9c5a82d`. Its numbers
should not be used.
