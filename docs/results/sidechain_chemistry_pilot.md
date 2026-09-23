# Side-chain chemistry on the three integrated cells, against a native baseline

GT-free metrics only. RMSD, chi-recovery and rotamer-recovery compare a side
chain to a NATIVE one and so need the native sequence; the design pipeline
emits a designed sequence, so there is no per-residue correspondence and
those metrics are undefined on its output. What survives is whether the
packing is internally sensible.

Reproduce: `python scripts/eval_sidechain_chemistry.py <cell-dir>`

## The native baseline comes first

Ten native target structures from the binder benchmark, identical metric:

| | native |
|---|---|
| chi1 outliers (>40 deg from nearest well) | **3.6 %** |
| side-chain clashes per 100 SC atoms | **2.19** (range 1.30-3.11) |

## The cells

| cell | chi1 outliers | clashes /100 SC atoms | vs native |
|---|---|---|---|
| `event_fixed`, 1 event | 0.0 % | **0.51** | **0.23x** |
| redecode, 1 event | 0.0 % | **2.90** | 1.32x |
| redecode, 4 events | 0.2 % | **6.35** | **2.90x** |

## Reading it, which is not the obvious way round

**A low clash count is not good.** `event_fixed` at 0.51 is a quarter of
native density. Real proteins pack tightly; a structure well under the
native rate is under-packed, not clean. So the monotone rise 0.51 -> 2.90 ->
6.35 is not simply "redecode degrades packing" -- the first step moves
packing from far below native to slightly above it, which is the direction
of a better-packed core, and only the four-event cell is genuinely
over-clashing at 2.9x native.

An earlier draft of this note read the same three numbers as a 6x
degradation caused by redecode. That was wrong for want of the baseline --
the same error as reading an all-atom interface minimum against a
backbone-only control.

**A 0 % chi1 outlier rate is not better than native either.** Real side
chains sit off-rotamer 3.6 % of the time; the packer essentially never
does. It is under-dispersed, placing every side chain in a canonical well.
That is a known conservative-packer signature, and it means these designs
are rotamerically *more* idealised than real protein, not more correct.

## What this does not establish

**n = 7 designs per cell, one target, one length, one generation seed.**
The native baseline is ten different proteins at 85-170 residues against
80-residue designed binders, so the comparison is indicative, not matched.

**It cannot rank the arms.** Within each cell the seven arms differ by less
than the spread between cells. This measures the protocol, not the adapter.

**It says nothing about whether the side chains are RIGHT**, only that they
are physically plausible and rotamerically canonical. Correctness needs a
native sequence, which is the separate packing benchmark
(`scripts/eval_couple.py`, held-out on the 31 val dimers -- cluster overlap
with the 512 J03/S03 training clusters is zero).

## Why this matters

`fold_af2ig.py` reads only CA/CB from the structure -- AF2 initial guess
takes `prev_pos` into `pseudo_beta_fn`, which is CB for everything except
glycine. **Designability is blind to every atom past CB.** All three cells
score 100 % designable while their clash density spans a 12x range. Any
claim about side-chain quality has to come from metrics like these, not
from the Table 4 conjunction.
